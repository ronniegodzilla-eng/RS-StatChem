"""DICOM reading with pydicom (MIT), following ImageJ 1.54 ``ij/plugin/DICOM.java`` (public
domain, NIH) for what the image becomes:

* 8/16-bit grey, signed 16-bit as ImageJ's signed images (raw = v + 32768, calibration
  ``-32768*slope + intercept``), RGB, multi-frame files as stacks;
* Rescale Slope != 1 opens as 32-bit with the slope and intercept applied (ImageJ's
  "open as float" path); otherwise a non-zero Rescale Intercept becomes a straight-line value
  calibration;
* Pixel Spacing (row\\column) -> pixel height/width in mm, Slice Thickness / Spacing Between
  Slices -> voxel depth; Window Center/Width -> display range;
* the header as the Info property, one line per element: ``0028,0030  Pixel Spacing: 0.5\\0.5``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rsstatchem.core.imaging.calibration import STRAIGHT_LINE
from rsstatchem.core.imaging.image import GRAY16, ImageDoc


def is_dicom(path) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(132)
    except OSError:
        return False
    return len(head) == 132 and head[128:132] == b"DICM"


def _value_text(elem) -> str:
    v = elem.value
    if elem.VR == "SQ":
        return ""
    if isinstance(v, (bytes, bytearray)):
        return "" if len(v) > 64 else v.decode("latin-1", "replace").strip("\0 ")
    if hasattr(v, "__iter__") and not isinstance(v, str):
        return "\\".join(str(x) for x in v)
    return str(v).strip()


def dicom_info(ds) -> str:
    """ImageJ-style header listing ("gggg,eeee  Name: value")."""
    lines = []

    def walk(dataset):
        for elem in dataset:
            if elem.tag == 0x7FE00010:
                continue
            name = elem.name if elem.name and not elem.name.startswith("[") else "---"
            text = _value_text(elem)
            lines.append(f"{elem.tag.group:04X},{elem.tag.element:04X}  {name}: {text}")
    if getattr(ds, "file_meta", None) is not None:
        walk(ds.file_meta)
    walk(ds)
    return "\n".join(lines) + "\n"


def read_dicom(path) -> ImageDoc:
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    if "PixelData" not in ds:
        raise ValueError("Unable to decode DICOM header (no pixel data).")
    arr = ds.pixel_array
    title = Path(path).name
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if spp == 3:
        if photometric.startswith("YBR"):
            from pydicom.pixels import convert_color_space
            arr = convert_color_space(arr, photometric, "RGB")
        axes = "ZYXS" if frames > 1 else "YXS"
        doc = ImageDoc.from_array(arr.astype(np.uint8), axes, title)
    else:
        axes = "ZYX" if frames > 1 else "YX"
        signed = int(getattr(ds, "PixelRepresentation", 0) or 0) == 1
        bits = int(getattr(ds, "BitsAllocated", 16) or 16)
        if slope != 1.0:
            data = arr.astype(np.float32) * np.float32(slope) + np.float32(intercept)
            doc = ImageDoc.from_array(data, axes, title)
        elif bits == 8 and not signed:
            doc = ImageDoc.from_array(arr.astype(np.uint8), axes, title)
            if intercept != 0.0:
                doc.calibration.function = STRAIGHT_LINE
                doc.calibration.coefficients = (intercept, slope)
        elif signed:
            doc = ImageDoc.from_array(arr.astype(np.int16), axes, title)   # signed16 cal
            if intercept != 0.0:
                doc.calibration.coefficients = (slope * -32768 + intercept, slope)
        elif bits > 16:
            doc = ImageDoc.from_array(arr.astype(np.float32), axes, title)
        else:
            doc = ImageDoc.from_array(arr.astype(np.uint16), axes, title)
            if intercept != 0.0:
                doc.calibration.function = STRAIGHT_LINE
                doc.calibration.coefficients = (intercept, slope)
        if photometric == "MONOCHROME1" and doc.bit_depth == 8:
            from rsstatchem.core.imaging.lut import LUT
            doc.luts = [LUT.grays().inverted()]
    cal = doc.calibration
    spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
    if spacing is not None and len(spacing) >= 2 and float(spacing[0]) and float(spacing[1]):
        cal.pixel_height, cal.pixel_width = float(spacing[0]), float(spacing[1])
        cal.unit = "mm"
        cal.pixel_depth = 1.0
    depth = getattr(ds, "SpacingBetweenSlices", None) or getattr(ds, "SliceThickness", None)
    if depth:
        try:
            cal.pixel_depth = float(depth)
        except (TypeError, ValueError):
            pass
    doc.reset_display_range()
    center, width = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    if center is not None and width is not None and not doc.is_rgb:
        c = float(center[0] if hasattr(center, "__len__") and not isinstance(center, str)
                  else center)
        w = float(width[0] if hasattr(width, "__len__") and not isinstance(width, str)
                  else width)
        if w > 0:
            lo, hi = c - w / 2, c + w / 2
            if doc.type != 2:
                lo, hi = cal.raw_value(lo, doc.bit_depth), cal.raw_value(hi, doc.bit_depth)
            doc.display_ranges = [(float(lo), float(hi))] * doc.n_channels
    doc.info = dicom_info(ds)
    if frames > 1 and doc.type == GRAY16:
        pass
    return doc


def write_dicom(doc: ImageDoc, path, uid: str | None = None) -> None:
    """Minimal secondary-capture DICOM of the image (8/16-bit grey or RGB; stacks as
    multi-frame), used by tests and the DICOM round trip (ImageJ itself has no DICOM writer)."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = uid or generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "OT"
    ds.PatientName = "Anonymous"
    ds.Rows, ds.Columns = doc.height, doc.width
    cal = doc.calibration
    if cal.scaled():
        ds.PixelSpacing = [f"{cal.pixel_height:g}", f"{cal.pixel_width:g}"]
    if doc.is_rgb:
        data = doc.pixels.reshape((-1, doc.height, doc.width, 3))
        ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
        ds.BitsAllocated = ds.BitsStored = 8
        ds.HighBit, ds.PixelRepresentation = 7, 0
    else:
        data = doc.pixels.reshape((-1, doc.height, doc.width))
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        if data.dtype == np.uint8:
            ds.BitsAllocated = ds.BitsStored = 8
            ds.HighBit = 7
        else:
            data = data.astype(np.uint16)
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
        ds.PixelRepresentation = 0
        if cal.calibrated() and len(cal.coefficients) >= 2:
            ds.RescaleIntercept = f"{cal.coefficients[0]:g}"
            ds.RescaleSlope = f"{cal.coefficients[1]:g}"
    if len(data) > 1:
        ds.NumberOfFrames = len(data)
    else:
        data = data[0]
    ds.PixelData = np.ascontiguousarray(data).tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    del pydicom
