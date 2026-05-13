import logging.config


def setupLogging():

    sLOG_LEVEL = "DEBUG"

    oLOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,  # prevents Uvicorn or other libraries from silencing the logs in the code
        "formatters": {
            "simple": {
                "format": "[%(levelname)s] [%(module)s] %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": sLOG_LEVEL,
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            # Root Logger settings
            "": {
                "handlers": ["console"],
                "level": sLOG_LEVEL,
            },
            "rio_tiler": { "level": "WARNING" },
            "rasterio": { "level": "WARNING" },
            "boto3": { "level": "WARNING" },
            "botocore": { "level": "WARNING" },
            "passlib": { "level": "WARNING" },
            "numexpr": { "level": "WARNING" },
            "fiona": { "level": "WARNING" },
            "pdfminer": { "level": "WARNING" },
            "unstructured": { "level": "WARNING" },
            "psparser": { "level": "WARNING" },
            "pdfinterp": { "level": "WARNING" },
            "cmapdb": { "level": "WARNING" },
            "numba": { "level": "WARNING" }, 
            "pdfminer": { "level": "WARNING" },
            "unstructured": { "level": "WARNING" },
            "sentence_transformers": { "level": "WARNING" },
            "transformers": { "level": "WARNING" },
            "huggingface_hub": { "level": "WARNING" },
            "httpx": { "level": "WARNING" },
            "httpcore": { "level": "WARNING" },
        },
    }

    logging.config.dictConfig(oLOGGING_CONFIG)