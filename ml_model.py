from copy import deepcopy
from math import *
from functions import *
import tensorflow as tf

#print(tf.zeros(2,2,2))
# testEntry = Entry(1, 2, 2, 2, 3, 3, 3)
# print(testEntry.acc)

input("Press enter to continue 1")

filePath = "arduino-data/falling/Venkat_falling_5_9_23.txt"

def make_markers(fileName):
    """return a list of the locations of all new trials"""
    markers = []
    with open(fileName, "r") as file:
        lines = file.readlines()
        print(lines[:10])
        for i in range(len(lines)):
            if lines[i] == 'New Trial\n':
                markers.append(i)
    return markers

# print(markers)

def get_entries(fileName):
    """Parses text file into a list of entries. Returns a list[Entry] object."""
    entries = []
    with open(fileName, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i] == ' + New event: \n':
                # try:
                #     entry = Entry(cnvd(lines[i + 1], i + 1), cnvd(lines[i + 2], i + 2), cnvd(lines[i + 3], i + 3), cnvd(lines[i + 4], i + 4), cnvd(lines[i + 5], i + 5), cnvd(lines[i + 6], i + 6), cnvd(lines[i + 7], i + 7), line=i)
                #     entries.append(entry)
                # except TypeError:
                #     print(f"There was a problem making the Entry at line {i}.")
                try:
                    entry = Entry(cnv(lines[i + 1]) / 1000, cnv(lines[i + 2]), cnv(lines[i + 3]), cnv(lines[i + 4]), cnv(lines[i + 5]), cnv(lines[i + 6]), cnv(lines[i + 7]), line=i)
                    entries.append(entry)
                except ValueError:
                    print(f"Error occurred in the entry from line {i}.")
    return entries



def get_organized_entries(entries: list[Entry], markers: list[int]) -> list[list[Entry]]:
    """Breaks up entries by trial"""
    organized_entries = []
    for i in range(len(markers) - 1):
        trial_entries = []
        for entry in entries:
            if entry.is_between(markers[i], markers[i + 1]):
                trial_entries.append(entry)
        organized_entries.append(trial_entries)
    return organized_entries


def get_batches(organized_entries: list[list[Entry]], batch_size = 15) -> list[list[Batch]]:
    batches = []
    for trial_entries in organized_entries:
        trial_batches = []
        if len(trial_entries) < batch_size:
            continue
        for i in range(len(trial_entries) - batch_size):
            trial_batch = Batch(trial_entries[i: i + batch_size], 0, first_line=trial_entries[i].line)
            trial_batches.append(trial_batch)
        batches.append(trial_batches)
    return batches





#GOAL: edit all elements in batches so that object.falling accurately represents if they're actually falling

#Attempt 1: check if all accelerations are greater than a certain number. If one is less, then it is falling
counter = 0
total_counter = 0
def make_simple_algo(threshold = 6.5):
    for trial_batches in batches:
        for batch in trial_batches:
            for entry in batch.entries:
                total_counter += 1
                if entry.acc < threshold:
                    counter += 1
                    #print(batch.falling)
                    batch.activate()
                    #print(batch.falling)
    print(counter)
    print(total_counter)



                    
markers = make_markers(filePath)
raw_entries = get_entries(filePath)
input("press enter to continue 2")
#print(raw_entries)
organized_entries = get_organized_entries(raw_entries, markers)
batches = get_batches(organized_entries)
input('press enter to continue 3')
threshold = 6.5

def activate_batches(batches: list[list[Batch]]):
    for trial_batches in batches:
        min_time = trial_batches[0].entries[0].time
        min_acc = trial_batches[0].entries[0].acc
        for batch in trial_batches:
            for entry in batch.entries:
                if entry.acc < min_acc:
                    min_acc = entry.acc
                    min_time = entry.time

        for batch in trial_batches:
            if min_time in batch.times:
                batch.activate()

def test_batches():
    input('press enter to continue 4')
    for sub_batch in batches:
        for batch in sub_batch:
            #print(batch.falling)
            if batch.falling == 1:
                print("you fell")
                print(batch.__str__())


activate_batches(batches)
test_batches()

#make_simple_algo()


# ML CODE Structure

# GENERAL ML CODE
#Turn batches into a usable format

#[[batch.time, batch.acc, ..., batch.falling][][][]]
# markers = make_markers(filePath)
# raw_entries = get_entries(filePath)
# organized_entries = get_organized_entries(raw_entries, markers)
# batches = get_batches(organized_entries)
# data_for_model_x = [[]]
# data_for_model_y = [[]]
# for i in range(len(batches)):
#     for batch in batches[i]:
#         ml_row = []
#         for entry in batch.entries:
#             ml_row += entry.get_arr()
#         data_for_model_x[i].append(ml_row)
#         data_for_model_y[i].append(batch.falling)

# #Split the data into test and train
# entries_for_test = floor(counter * 0.8)
# train_x = data_for_model_x[:entries_for_test]
# train_y = data_for_model_y[:entries_for_test]
# test_x = data_for_model_x[entries_for_test:]
# test_y = data_for_model_y[entries_for_test:]
#
#


#Make some structure
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Input(shape=(len(train_x[0]),)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='mean_squared_error')


#Train the model
#[e1, e2, e3...] = x

#model.fit(train_x, train_y, epochs=20)

#test the model
# model.evaluate(test_x, test_y)


# goodEntry = Entry(32, 312, 312, 543, 123, 54, 123)

# print(goodEntry.line)
# print(TypeError)
# badEntry = Entry(32)


# with open(r'arduino-data\falling\FallOnBedArduino.TXT', 'r') as f:
#     lines = f.readlines()



# def make_markers(fileName):
#     times = []
#     accelerations = []
#     rotations = []
#     markers = []
#     with open(fileName) as f:
#         lines = f.readlines()
#         i = 0
#         for line in lines:
#             if "New Trial" in line:
#                 markers.append(i)
#                 # print(i)
#             i += 1
#     return markers




        # lines = f.readlines()
        # j = 1
        # for i in range(1, len(lines)):
        #     if "New Trial" in lines[i]:
        #         times.append(temp_times)
        #         accelerations.append(temp_accelerations)
        #         rotations.append(temp_rotations)
        #         s = simpleFallingWithoutFile(temp_times, temp_accelerations)
        #         # global_minimum(temp_times, temp_accelerations)
        #         # print(f"falling of file {j}: {s}")
        #         temp_times.clear()
        #         temp_accelerations.clear()
        #         temp_rotations.clear()
        #         j += 1
        #     if "New event" in lines[i]:
        #         try:
        #             acc_x = float(lines[i+2])
        #             acc_y = float(lines[i+3].split("\n")[0])
        #             acc_z = float(lines[i+4])
        #             rot_x, rot_y, rot_z = float(lines[i+5]), float(lines[i+6]), float(lines[i+7])
        #             temp_rotations.append(comp(rot_x, rot_y, rot_z))
        #             temp_times.append(float(lines[i+1]) / 1000)
        #             temp_accelerations.append(comp(acc_x, acc_y, acc_z))
        #         except:
        #             print("You tried, it's ok!")
        #             print(i)
        # return [times, times, times]
    

