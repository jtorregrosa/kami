"""kami: rent a GPU only while you use it, to turn images into 3D meshes and prompts into images."""

__version__ = "0.1.0"

# RunPod's proxy (Cloudflare) returns 403 to urllib's default User-Agent.
USER_AGENT = f"kami/{__version__}"
