import logging
import sys

config = {
    'version': 1,
    'loggers': {
        'root': {
            'level': logging.INFO,
            'handlers': ['console']
        },
        'console': {
            'level': logging.INFO,
            'handlers': ['console'],
            'propagate': False
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': logging.DEBUG,
            'formatter': 'default',
            'stream': sys.stdout,
        }
    },
    'formatters': {
        'default': {
            'format': '%(asctime)s %(levelname)s %(module)s | %(message)s'
        }
    }
}
