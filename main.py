from util import *
from ultralytics import YOLO

detector = YOLO('Model/Best_Person_Detection.pt')
feature_extractor = tf.keras.models.load_model('Model/feature_extractor')

# classes = [0,2,13]
# for cls in classes:
#     add_new_person(cls, detector, feature_extractor, 0)

recognition(detector, feature_extractor)


# while True:
#     print('\n')
#     print('0. End \n')
#     print('1. Add new person \n')
#     print('2. Show attended \n')
#     print('3. Attendance \n')



#     select = input('Enter selection from (0-3): ')
#     if select == '0':
#         break
#     elif select == '1':
#         name = input('Enter name: ')
#         mode = input('Enter 0 (camera) - file_video: ')
#         add_new_person(name, detector, feature_extractor, mode)
#     elif select == '2':
#         path = get_path()
#         atd_path = os.path.join(path, 'attended_table.csv')
#         show_atd(atd_path)
#     elif select == '3':
#         recognition(detector, feature_extractor)
#     else:
#         print('Please selection from (0-3)')