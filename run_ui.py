import uvicorn

import ui  # noqa: F401
from chainlit.server import app


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
