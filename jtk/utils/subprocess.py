import os
import shutil
import subprocess
import sys
import signal
import re
from typing import List, Union


class Jumanpp:
    def __init__(self,
                 command='jumanpp',
                 timeout=30,
                 segmentaion=True,
                 option='',
                 rcfile='',
                 ignorepattern='',
                 pattern=r'^EOS$',
                 ):
        self.command = command
        self.options = option.split()
        self.timeout = timeout
        self.rcfile = rcfile
        self.ignorepattern = ignorepattern
        self.pattern = pattern

        cmds = [self.command] + self.options
        if self.rcfile:
            cmds += ['-r', self.rcfile]
        if segmentaion:
            cmds += ["--segment"]

        if shutil.which(command):
            self.analyzer = Subprocess(command=cmds, timeout=timeout)
        else:
            raise Exception(f"Can't find {command}! Make sure jumanpp (https://github.com/ku-nlp/jumanpp) is installed。")

        if self.rcfile and not os.path.isfile(os.path.expanduser(self.rcfile)):
            raise Exception("Can't read rcfile (%s)!" % self.rcfile)

    def tokenize(self, input_sent: Union[str, List[str]]):
        if isinstance(input_sent, str):
            res = self.analyzer.query(input_sent, pattern=r"\n$")
            return res
        elif isinstance(input_sent, list):
            res = []
            res = [self.analyzer.query(s, pattern=r"\n$") for s in input_sent]
            return res
        else:
            raise Exception("Unexpected input sentence!")


class Subprocess:
    def __init__(self, command, timeout=180):
        subproc_args = {'stdin': subprocess.PIPE, 'stdout': subprocess.PIPE,
                        'cwd': '.', 'close_fds': sys.platform != 'win32'}
        try:
            env = os.environ.copy()
            self.process = subprocess.Popen(command, env=env, **subproc_args)
            self.process_command = command
            self.process_timeout = timeout
        except OSError:
            raise

    def __del__(self):
        self.process.stdin.close()
        self.process.stdout.close()
        try:
            self.process.kill()
            self.process.wait()
        except OSError:
            # rased when trying to execute a non-existent file
            pass
        except TypeError:
            pass
        except AttributeError:
            pass

    def query(self, sentence: str, pattern: str):
        sentence = sentence.strip() + '\n'

        def alarm_handler(sifnum, frame):
            raise subprocess.TimeoutExpired(self.process_command, self.process_timeout)

        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(self.process_timeout)
        result = ''
        try:
            self.process.stdin.write(sentence.encode('utf-8'))
            self.process.stdin.flush()
            while True:
                if self.process.poll() is not None:
                    break
                line = self.process.stdout.readline().decode('utf-8').rstrip()
                if re.search(pattern, line):
                    break
                result += line + '\n'
        finally:
            signal.alarm(0)
        self.process.stdout.flush()
        return result