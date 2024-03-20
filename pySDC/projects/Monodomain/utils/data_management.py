import sqlite3
import json
import os


class database:
    def __init__(self, name):
        self.name = name
        path = os.path.dirname(self.name)
        if path != "" and not os.path.exists(path):
            os.makedirs(path)
        self.conn = sqlite3.connect(f'{self.name}.db')
        self.cursor = self.conn.cursor()

    # def write_arrays(self, table, arrays, columns_names=None):
    #     if not isinstance(arrays, list):
    #         arrays = [arrays]
    #     n = len(arrays)
    #     if columns_names is None:
    #         columns_names = ["val_" + str(i) for i in range(n)]

    #     self.cursor.execute(f'DROP TABLE IF EXISTS {table}')
    #     self.cursor.execute(f'CREATE TABLE {table} ({self._convert_list_str_to_arg(columns_names)})')
    #     self.cursor.execute(f'INSERT INTO {table} VALUES ({self._convert_list_str_to_arg(["?"]*n)})', arrays)
    #     self.conn.commit()

    def write_dictionary(self, table, dic):
        self.cursor.execute(f'DROP TABLE IF EXISTS {table}')
        self.cursor.execute(f'CREATE TABLE {table} (dic TEXT)')
        self.cursor.execute(f'INSERT INTO {table} VALUES (?)', [json.dumps(dic)])
        self.conn.commit()

    # def read_arrays(self, table, columns_names=None, dtype=np.double):
    #     if columns_names is None:
    #         self.cursor.execute(f"SELECT * FROM {table}")
    #     else:
    #         self.cursor.execute(f"SELECT {self._convert_list_str_to_arg(columns_names)} FROM {table}")
    #     result = self.cursor.fetchone()
    #     result_new = list()
    #     for res in result:
    #         result_new.append(np.frombuffer(res, dtype=dtype))
    #     if len(result_new) > 1:
    #         return result_new
    #     else:
    #         return result_new[0]

    def read_dictionary(self, table):
        self.cursor.execute(f"SELECT dic FROM {table}")
        (json_dic,) = self.cursor.fetchone()
        return json.loads(json_dic)

    def _convert_list_str_to_arg(self, str_list):
        return str(str_list).replace("'", "").replace("[", "").replace("]", "")

    def __del__(self):
        self.conn.close()


# def main():
#     data = database("test")
#     a = np.array([1.0, 2.0, 3.0])
#     b = np.array([10.0, 11.0, 12.0])
#     data.write_arrays("ab_table", [a, b], ['a', 'b'])
#     data.write_arrays("a_table", [a], ['a'])
#     data.write_arrays("b_table", b, 'b')
#     data.write_arrays("ab_table_noname", [a, b])
#     data.write_arrays("b_table_noname", b)

#     a_new, b_new = data.read_arrays("ab_table", ['a', 'b'])
#     print(a)
#     print(a_new)
#     print(b)
#     print(b_new)

#     a_new = data.read_arrays("a_table", "a")
#     print(a)
#     print(a_new)
#     b_new = data.read_arrays("b_table", "b")
#     print(b)
#     print(b_new)

#     a_new, b_new = data.read_arrays("ab_table_noname")
#     print(a)
#     print(a_new)
#     print(b)
#     print(b_new)

#     b_new = data.read_arrays("b_table_noname")
#     print(b)
#     print(b_new)

#     dic = {"name": "Giacomo", "age": 33}
#     data.write_dictionary("dic_table", dic)
#     dic_new = data.read_dictionary("dic_table")
#     print(dic)
#     print(dic_new)

#     data_read = database("test")
#     a_new, b_new = data_read.read_arrays("ab_table", ['a', 'b'])
#     print(a)
#     print(a_new)
#     print(b)
#     print(b_new)


# if __name__ == "__main__":
#     main()
