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

from datetime import datetime


class Timer(object):
    def __init__(self, description=None):
        self.description = description
        self.start_time = None
        self.milestones = []
        self.texts = []

    def start(self):
        self.start_time = datetime.now()
        self.milestones.append(self.start_time)
        self.texts.append('start')

    # unit can be min or sec
    def mark(self, text, unit='min'):
        assert(unit == 'sec' or unit == 'min')

        now = datetime.now()
        last = self.milestones[-1]
        self.milestones.append(now)
        self.texts.append(text)

        since_last = (now - last).total_seconds()
        since_start = (now - self.start_time).total_seconds()

        if unit == 'min':
            since_last /= 60.
            since_start /= 60.

        return now, since_last, since_start

    def summary(self):
        template = '%Y-%m-%d %H:%M:%S'
        end_time = self.milestones[-1]
        report = 'Timer started at {}, summarized at {}, total elapsed {:.6} minutes\n'.format(
            self.start_time.strftime(template), end_time.strftime(template), (end_time - self.start_time).total_seconds() / 60.
        )
        report += '\tdescription: {}\n'.format(self.description)
        report += '\tstart: {}\n'.format(self.start_time.strftime(template))
        cnt = len(self.milestones)
        for i in range(1, cnt):
            since_last = (self.milestones[i] - self.milestones[i-1]).total_seconds() / 60.
            since_start = (self.milestones[i] - self.start_time).total_seconds() / 60.

            report += '\t{}: {}, since_last: {:.6} minutes, since_start: {:.6} minutes\n'.format(
                self.texts[i], self.milestones[i].strftime(template), since_last, since_start)

        return report


if __name__ == '__main__':
    timer = Timer('my timer test')
    timer.start()

    import time
    time.sleep(5)
    timer.mark('just slept 5 seconds')

    time.sleep(8)
    timer.mark('just slept 8 seconds')

    print(timer.summary())
