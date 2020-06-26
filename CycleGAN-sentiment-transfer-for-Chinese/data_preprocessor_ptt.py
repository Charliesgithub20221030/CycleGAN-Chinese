# -*- coding: utf-8 -*
import jieba
import json
import re
import os

data_path = 'data'
ptt_dataset = os.path.join(data_path, 'PTT_dataset')

jieba.set_dictionary(os.path.join(data_path, 'dict.txt.big.txt'))
jieba.initialize()
dictionary = set()


def ptt_data_processor():
    global dictionary
    pattern_link = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    pattern_filter = r'[\n\x1b\xa0\{\}\[\]a-zA-Z0-9＊※【】＜︱＞●▃▂▄●▂▃ψ█＼●／●／︱＞《》+_\)\(*&^%$#@!~`=)|』<>『」:;「\.\-\'\"\\/¤§¨®¯°±·¸¼½¾¿ÀÄÆÉÒÓÕ×ØÙÜßàáâäåçèéêëìíîñòóö÷ùúüāēīōūǎǐǒǔǖʊˇˉˊˋˍ˙ΑΓΔΕΘΙΛΜΞΟΠΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχωАВГДЖЗИМНПСавдезиклнопрстуья–—‘’“”‡‥…‧‰′‵‼€℃℅℉№℡™⅓⅙⅚ⅠⅡⅢⅣⅥⅦⅩⅲⅴⅸ←↑→↓↔↕↖↗↘↙↨↲↸↹⇦⇧⇨⇩∀∕√∞∟∠∣∥∩∪∫∮∴∵∷≒≠≡≦≧⊂⊃⊇⊕⊙⊥⊿⌂⌒①⑤⑴⑶⑻⒀⒈⒉⒊⒋⒌⒍ⓝ─━│┃┌┐└┘├┤┬┴┼╎═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴▁▅▆▇▉▊▋▌▍▎▏▓▔▕■□▲△▶▼▽◄◆◇○◎◘◙◢◣◤◥☁☂★☆☎☑☒☛☜☞☟☠☯☰☱☲☳☴☵☶☷☹☺☻☼☽♀♂♠♡♣♤♥♦♧♪♬✂✈✉✓✕✡✩✽❏❶❺❻❾\u3000、。〃々〆〇〈〉〒〔〕〗〝〞〡〢〣〥〨ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわをん゛゜ゝゞァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチッツテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヱヲンヴヶ・ーヽヾㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩ㊣㎜㎡㏄䒑]'

    def get_content(data, positive=True):
        global dictionary
        if positive:
            label = '1'
        else:
            label = '0'
        content = []
        for d in data:
            line = d['e_內文'].replace(' ', '')
            line = re.sub(pattern_link, "", line)
            line = re.sub(pattern_filter, "", line)
            line = re.sub(r'[^\u4e00-\u9fff\.\?。，？!！]+', '', line)
            if line == '' or line == '\n':
                continue
            list_line = list(jieba.cut(line))

            # add word to dictionary set
            for word in list_line:
                dictionary.add(word)

            # retain space between each word(easyly to encode)
            content_line = ' '.join(list_line)
            line = label+" +++$+++ "+content_line
            content.append(line)
        return content

    total = []
    print('deal with gossiping1500.json ...')
    with open(os.path.join(ptt_dataset, 'Gossiping1500.json'), 'r') as f:
        pos_data = json.load(f)
    total = total + get_content(pos_data, True)
    print('deal with Pttprozac.json ...')
    with open(os.path.join(ptt_dataset, 'Pttprozac.json'), 'r') as f:
        neg_data = json.load(f)
    total = total + get_content(neg_data, False)
    print('deal with Hate.json ...')
    with open(os.path.join(ptt_dataset, 'Hate.json'), 'r') as f:
        neg_data = json.load(f)
    total = total + get_content(neg_data, False)

    with open(os.path.join(data_path, 'feature_ptt_spliit.txt'), 'w') as f:
        f.writelines('\n'.join(total))
    print('Done')


def export_dictionary():
    global dictionary
    dictionary_list = sorted(list(dictionary))
    dic = dict()
    dic['<PAD>'] = 0
    dic['<UNK>'] = 1
    dic['<EOS>'] = 2
    dic['<BOS>'] = 3
    for i in range(4, len(dictionary_list)+4):
        dic[dictionary_list[i-4]] = i

    with open('data/dictionary_ptt_split.json', 'w') as f:
        json.dump(dic, f)


def feature_to_dictionary():
    content = '''<PAD> 0\n<UNK> 1\n<EOS> 2\n<BOS> 3\n'''

    feature_file = os.path.join(data_path, 'feature_ptt.txt')
    with open(feature_file, 'r') as f:
        feature = f.readlines()
    feature_text = ''.join([f.split(' +++$+++ ')[1] for f in feature])
    chars = sorted(set(list(jieba.cut(feature_text))))
    chars = [c for c in chars if c != '\n' and c != '']
    print('chars : ', len(chars))

    for i, c in enumerate(chars):
        line = c+" "+str(i+4)+'\n'
        content = content + line

    with open(os.path.join(data_path, 'dictionary_ptt_split.txt'), 'w') as f:
        f.write(content)

    print('done, check dictionary_ptt.txt to see the result.')


def jsonify_dictionary():
    with open(os.path.join(data_path, 'dictionary_ptt.txt'), 'r') as f:
        data = f.readlines()

    result = dict()
    for line in data:
        k, v = line.replace('\n', '').split(' ')
        result[k] = v

    result_json = json.dumps(result, ensure_ascii=False)
    with open(os.path.join(data_path, 'dictionary_ptt_split.json'), 'w', encoding='utf-8') as f:
        f.write(result_json)


if __name__ == "__main__":
    print('processing ptt dataset...')
    ptt_data_processor()
    print('converting processed data to dictionary...')
    # feature_to_dictionary()
    print('converting dictionary to json format...')
    export_dictionary()
    # jsonify_dictionary()
    print('Done.')
