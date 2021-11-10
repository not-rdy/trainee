import pandas as pd
import re


class Check:

    def __init__(self):

        # типы ошибок/предупреждений
        self.messages = {'error_1': 'Область удалена',
                         'error_2': 'Изменена левая координата',
                         'error_3': 'Изменена правая координата',
                         'error_4': 'Комментарий не по формату',
                         'warning_1': 'Добавлена новая область'}

    def __append(self, typ, coord, mess):
        self.message_table = self.message_table.append(pd.DataFrame({'Type': [typ],
                                                                     'Coords': [coord],
                                                                     'Message': [mess]}))

    def checkup(self, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:

        self.message_table = pd.DataFrame({'Type': [], 'Coords': [], 'Message': []})

        # ПРОВЕРКА:
        for row_b in range(len(before)):
            for row_a in range(len(after)):

                # 1) поиск измененных значений координат в after_feedback
                if after.loc[row_a, 'thread_num'] == before.loc[row_b, 'thread_num'] \
                    and (after.loc[row_a, 'start_coord'] != before.loc[row_b, 'start_coord']
                         and after.loc[row_a, 'end_coord'] == before.loc[row_b, 'end_coord']):
                    typ = 'error!'
                    coord = (after.loc[row_a, 'thread_num'],
                             after.loc[row_a, 'start_coord'],
                             after.loc[row_a, 'end_coord'])
                    mess = self.messages['error_2']
                    self.__append(typ, coord, mess)

                elif after.loc[row_a, 'thread_num'] == before.loc[row_b, 'thread_num'] \
                        and (after.loc[row_a, 'start_coord'] == before.loc[row_b, 'start_coord']
                             and after.loc[row_a, 'end_coord'] != before.loc[row_b, 'end_coord']):
                    typ = 'error!'
                    coord = (after.loc[row_a, 'thread_num'],
                             after.loc[row_a, 'start_coord'],
                             after.loc[row_a, 'end_coord'])
                    mess = self.messages['error_3']
                    self.__append(typ, coord, mess)

        # поиск удаленных строк
        for row in range(len(before)):
            if (before.loc[row, 'start_coord'] not in after.loc[:, 'start_coord'].values) \
                    and (before.loc[row, 'end_coord'] not in after.loc[:, 'end_coord'].values):
                typ = 'error!'
                coord = (before.loc[row, 'start_coord'],
                         before.loc[row, 'end_coord'])
                mess = self.messages['error_1']
                self.__append(typ, coord, mess)

        # поиск добавленных строк
        for row in range(len(after)):
            if (after.loc[row, 'start_coord'] not in before.loc[:, 'start_coord'].values) \
                    and (after.loc[row, 'end_coord'] not in before.loc[:, 'end_coord'].values):
                typ = 'warning!'
                coord = (after.loc[row, 'start_coord'], after.loc[row, 'end_coord'])
                mess = self.messages['warning_1']
                self.__append(typ, coord, mess)

        # проверка комментария на соответствие формату
        for i in range(len(after)):
            right_content = after.loc[i, 'right_border_content']

            if pd.isna(right_content):
                typ = 'error!'
                coord = (after.loc[i, 'thread_num'],
                         after.loc[i, 'start_coord'],
                         after.loc[i, 'end_coord'])
                mess = self.messages['error_4']
                self.__append(typ, coord, mess)

            else:
                right_content = right_content.split('_')

                if len(right_content) != 3:
                    typ = 'error!'
                    coord = (after.loc[i, 'thread_num'],
                             after.loc[i, 'start_coord'],
                             after.loc[i, 'end_coord'])
                    mess = self.messages['error_4']
                    self.__append(typ, coord, mess)

                elif len(right_content) == 3:
                    yes_no = right_content[0]
                    if yes_no.lower() not in ['да', 'нет', 'пропуск']:
                        typ = 'error!'
                        coord = (after.loc[i, 'thread_num'],
                                 after.loc[i, 'start_coord'],
                                 after.loc[i, 'end_coord'])
                        mess = self.messages['error_4']
                        self.__append(typ, coord, mess)
                    else:
                        pass

                    punctuations = ['.', ',', ';', ':']
                    channels = re.sub(r'\s', '', right_content[1])
                    for punct in punctuations:
                        channels = channels.replace(punct, '')
                    if not channels.isdigit():
                        typ = 'error!'
                        coord = (after.loc[i, 'thread_num'],
                                 after.loc[i, 'start_coord'],
                                 after.loc[i, 'end_coord'])
                        mess = self.messages['error_4']
                        self.__append(typ, coord, mess)
                    else:
                        pass

        return self.message_table
