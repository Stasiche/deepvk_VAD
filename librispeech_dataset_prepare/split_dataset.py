import random
import csv

split_rate = 0.3
with open('activity_train.csv', 'w') as train_out:
    with open('activity_val.csv', 'w') as val_out:
        with open('activity.csv', 'r') as f:
            reader = csv.reader(f)
            train_writer = csv.writer(train_out)
            val_writer = csv.writer(val_out)
            for row in reader:
                if random.random() >= split_rate:
                    train_writer.writerow(row)
                else:
                    val_writer.writerow(row)
