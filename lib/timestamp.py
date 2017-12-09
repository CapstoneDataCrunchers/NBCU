import pickle

file = open('scripts.json','rb')
json1 = pickle.load(file)

topic_word = input('Please type in your topic word ')
print('\n','Episodes whose scripts contain the topic word: ', topic_word,'\n')
index = 1
for episode in json1:
    print('episode:', index)
    line_index = 0
    for script in episode:
        try:
            line = script['line']
            if topic_word in line:
                time_stamps1 = episode[line_index-1]['time']
                time_stamps2 = episode[line_index+1]['time']
                
                if time_stamps1[0] == '\ufeff01':
                    time_stamps1 = time_stamps1[1:]
                if time_stamps2[0] == '\ufeff01':
                    time_stamps2 = time_stamps2[1:]
                    
                time_stamp1 = time_stamps1.split(',')
                time_stamp2 = time_stamps2.split(',')

                
                time1 = time_stamp1[0].split(':')
                time2 = time_stamp2[1].split(':')

                try:
                    hour1   = float(time1[0])
                    minute1 = float(time1[1])
                    second1 = float(time1[2])
                except:
                    hour1   = float(time1[0][1:])
                    minute1 = float(time1[1][1:])
                    second1 = float(time1[2][1:])

                hour2   = float(time2[0])
                minute2 = float(time2[1])
                second2 = float(time2[2])

                time = (hour2 - hour1) * 3600 + (minute2 - minute1) * 60 + (second2 - second1)
                
                print(episode[line_index-1]['line'])
                print(line)
                print(episode[line_index+1]['line'])
                print(time_stamp1[0],":", time_stamp2[1])
                print('lasting time:', time,'s', '\n')
        except:
            pass
        line_index += 1
    print(' ')
    index += 1