import os
from dotenv import load_dotenv

load_dotenv()

LOCAL_MODEL_STORAGE = os.environ.get('LOCAL_MODEL_STORAGE', None)

# DEFAULT_MODEL_PATH = 