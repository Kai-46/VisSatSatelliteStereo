#  ===============================================================================================================
#  Copyright (c) 2019, Cornell University. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright otice, this list of conditions and
#        the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#        the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#      * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
#        promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#  OF SUCH DAMAGE.
#
#  Author: Kai Zhang (kz298@cornell.edu)
#
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
#  ===============================================================================================================


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
