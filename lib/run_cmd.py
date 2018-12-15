import logging
import subprocess
import shlex


# redirect stdout, stderr to file
def run_cmd(cmd):
    logging.info('Running subprocess: {}'.format(cmd))

    try:
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # output, _ = process.communicate()
        # output = output.decode('ascii')
        # logging.info(output)
        #print('output: {}'.format(output))

        # interacting with long-running subprocess
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logging.info(output)
    except (OSError, subprocess.CalledProcessError) as exception:
        logging.error('Exception occured: {}'.format(exception))
        logging.error('Subprocess failed')
        exit(-1)
    else:
        # no exception was raised
        logging.info('Subprocess finished')
