import csv, sys
import os
import wave
import contextlib

filename = 'AudioSet/balanced_train_segments.csv'
rownum = 0
path = 'AudioSet/balanced_train/'
project_path = '/home/joseph/PycharmProjects/project/src/'

def youtube_download_os_call(id, start_time, idx):
    ret = os.system('ffmpeg -ss ' + start_time +
              ' -i $(youtube-dl --extract-audio '
              '--audio-format wav --audio-quality 0 '
              '--get-url https://www.youtube.com/watch?v=' + id + ')'
              ' -t 10 ' + path + idx + '_' + id + '.wav')

    return ret

def get_wav_file_length(path, idx, id):
    sample = project_path + path + idx + '_' + id + '.wav'
    with contextlib.closing(wave.open(sample, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        length = frames / float(rate)
        print(length)

    return length

def create_error_file(id, idx):
    with open(path + idx + '_' + id + '_ERROR.wav', 'a'):
        os.utime(path + idx + '_' + id + '_ERROR.wav', None)

def youtube_downloader(id, start_time, idx):
    ret = youtube_download_os_call(id, start_time, idx)

    print('ffmpeg -ss ' + start_time +
              ' -i $(youtube-dl --extract-audio '
              '--audio-format wav --audio-quality 0 '
              '--get-url https://www.youtube.com/watch?v=' + id + ')'
              ' -t 10 AudioSet/balanced_train/' + idx + '_' + id + '.wav')
    return ret

with open(filename, newline='') as f:
    reader = csv.reader(f)
    try:
        for row in reader:
            # Skip the 3 line header
            if rownum >= 3:
                print(row)
                ret = youtube_downloader(row[0], str(float(row[1].lstrip())),
                                   str(rownum - 3))
                # If there was an error downloading the file
                # This sometimes happens if videos are blocked or taken down
                if ret != 0:
                    create_error_file(row[0], str(rownum - 3))

            rownum += 1
    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
