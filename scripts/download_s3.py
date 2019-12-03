import sys
import subprocess
import rlkit.launchers.config as config

cmd = F"aws s3 sync --exact-timestamp --exclude '*' --include '12-02*' {config.AWS_S3_PATH}/ ../../s3_files/"

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
print(cmd)
for line in iter(process.stdout.readline, b''):
    sys.stdout.buffer.write(line)