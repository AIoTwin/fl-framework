"""
TODO: Some generic Thread and process pool handler
"""
import subprocess

from log_infra import def_logger
from threading import Thread

logger = def_logger.getChild(__name__)

CLIENT_EXEC = "fl_common/clients/client_exec.py"
# SERVER_EXEC = "fl_common/servers/server_exec.py"
AGGREGATOR_EXEC = "fl_common/aggregators/aggregator_exec.py"


class ThreadWrapper(Thread):
    """
        Use this for lazy starting startable FL entities
    """

    def __init__(self, runnable_fl_entity):
        Thread.__init__(self)
        self.runnable_fl_entity = runnable_fl_entity

    def run(self):
        self.runnable_fl_entity.start()


# we don't need no security
def run_script(path: str, args, std_stream=None, err_stream=None):
    """
        subprocess.PIPE to create separate accessable streams
    """
    cmd = ["python3", path] + args
    result = subprocess.run(cmd, stdout=std_stream, stderr=err_stream)
    return result
