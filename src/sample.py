# blackがちゃんと動いているか確認するためのサンプルコード
# 1行で書いても88行以内に収まるように整形される
def tooooooooooooooooo_long_function_name(
    toooooooooooooooo_long_arg1, tooooooooooooo_long_arg2
):

    return None


# mypyの型チェックが入るか確認するためのサンプルコード
# function is missing a type annotationみたいなエラーが出る
def add_numbers(a, b):
    return a + b


# こっちは型チェックが通るはず
def sub_numbers(a: int, b: int) -> int:
    return a - b


# allow-redefinitionを設定しているので、再定義できる
def convert_number_to_string(number: int) -> str:
    number = str(number)
    return number


def multiply_numbers(a: int, b: int) -> int:
    return a * b


def divide_numbers(a: int, b: int) -> float:
    return a / b


if __name__ == "__main__":
    print(add_numbers(1, 2))
    print(sub_numbers(1, 2))
    print(convert_number_to_string(123))
