# ===============================================================================================================
#  This file is part of Creation of Operationally Realistic 3D Environment (CORE3D).                            =
#  Copyright 2019 Cornell University - All Rights Reserved                                                      =
#  -                                                                                                            =
#  NOTICE: All information contained herein is, and remains the property of General Electric Company            =
#  and its suppliers, if any. The intellectual and technical concepts contained herein are proprietary          =
#  to General Electric Company and its suppliers and may be covered by U.S. and Foreign Patents, patents        =
#  in process, and are protected by trade secret or copyright law. Dissemination of this information or         =
#  reproduction of this material is strictly forbidden unless prior written permission is obtained              =
#  from General Electric Company.                                                                               =
#  -                                                                                                            =
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),     =
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.            =
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes. =
# ===============================================================================================================

import logging
import subprocess
import shlex


# redirect stdout, stderr to file
def run_cmd(cmd, disable_log=False, input=None):
    if not disable_log:
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
        logging.error('Exception occured: {}, cmd: {}'.format(exception, cmd))
        logging.error('Subprocess failed')
        exit(-1)
    else:
        if not disable_log:
            # no exception was raised
            logging.info('Subprocess finished')
