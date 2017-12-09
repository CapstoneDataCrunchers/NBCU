import pickle

'''This script is for finding the episodes that occur particular Topic Word(e.g. Coffee, Wine, Basketball),
   And the correponding clips (e.g. 01:02:48 -- 01:02:51), and lasting time   
'''
def loading_data():
    json1 = pickle.load(open('scripts.json','rb'))
    return json1

def query():
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
                    time_stamps = script['time']
                    if time_stamps[0] == '\ufeff01':
                        time_stamps = time_stamps[1:]
                    time_stamp = time_stamps.split(',')

                    time1 = time_stamp[0].split(':')
                    time2 = time_stamp[1].split(':')

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
                    print(time_stamps)
                    print('lasting time:', time,'s', '\n')
            except:
                pass
            line_index += 1
        print(' ')
        index += 1
        
if __name__ == '__main__':
    json1 = loading_data()
    query()
       
