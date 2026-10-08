import importlib

import uvicorn

from chainlit.server import app

importlib.import_module("ui")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
