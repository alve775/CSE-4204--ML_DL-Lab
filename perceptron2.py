import random

def create_dataset():
    class0 = []
    class1 = []

    for i in range(16):
        bits = format(i, "04b")
        x  = [int(bit) for bit in bits]

        if i < 8:
            y = 0
            class0.append(( x, y))

        else:
            y = 1
            class1.append(( x, y))

        return class0, class1


def split_dataset(class0, class1, split_number):
    random.shuffle(class0)
    random.shuffle(class1)

    train_count_per_class = round((split_number / 10) * 8)




if __name__ == "__main__":
    class0, class1 = create_dataset()
    print(class0)
    print(class1)