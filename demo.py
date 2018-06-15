import os

import predict


def main():
    PATH = os.getcwd()

    print(predict.predict(PATH + '/test/faceb00k.bid..screen.png', PATH + '/test/faceb00k.bid..source.txt'))


if __name__ == '__main__':
    main()