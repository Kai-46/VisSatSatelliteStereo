import logging
import subprocess
import shlex


# redirect stdout, stderr to file
def run_cmd(cmd, disable_log=False, input=None):
    logging.info('Running subprocess: {}'.format(cmd))

    try:
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)

        if input is not None:
            # interacting with short-running subprocess
            output = process.communicate(input=input.encode())[0]
            if not disable_log:
                logging.info(output.decode())
            else:
                process.wait()
        else:
            # interacting with long-running subprocess
            if not disable_log:
                while True:
                    output = process.stdout.readline().decode()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        logging.info(output)
            else:
                process.wait()
    except (OSError, subprocess.CalledProcessError) as exception:
        print('oh my goodness!')
        logging.error('Exception occured: {}'.format(exception))
        logging.error('Subprocess failed')
        exit(-1)
    else:
        # no exception was raised
        logging.info('Subprocess finished')
