"""RS-StatChem connector: a local MCP (Model Context Protocol) server.

``RS-StatChem --mcp`` serves the whole application over stdio so that an MCP client the user
already has (the Claude desktop app, Claude Code) can work with it. The app never signs in to
anything: the client is the one talking to Claude. See docs/DEVELOPMENT.md, "Claude connector".
"""
