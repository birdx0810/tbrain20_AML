import transformers
import sys


def map_unk(sentence, tokens, ids, unk_token_id, special_tokens):
    """
    If id([UNK]) = 100 and ids = [9, 100, 27], then the result mapping would be [token(9), token(100), token(27)].
    len(mapping) = len(ids)

    Returns:
        `mapping`: [str, str, str, ...]
    """

    mapping = []
    cur_sentence_index = 0
    for ids_index, cur_id in enumerate(ids):

        if ids_index < len(mapping):
            continue

        if cur_id != unk_token_id:
            cur_token = tokens[ids_index]
            mapping.append(cur_token)
            # update char index
            if cur_token not in special_tokens:
                cur_sentence_index += sentence[cur_sentence_index:].find(cur_token) + len(cur_token) # here: may be -1?
        else:
            # find next non-unk token
            next_non_unk_index = -1
            for temp_ids_index in range(ids_index+1, len(ids)):
                if ids[temp_ids_index] != unk_token_id:
                    next_non_unk_index = temp_ids_index
                    break

            next_non_unk_token = tokens[next_non_unk_index]
            
            # split unk string
            # TODO: check len(split_result) == len(unk)
            next_sentence_index = sentence[cur_sentence_index:].find(next_non_unk_token)
            if next_sentence_index == -1:
                string_to_split = sentence[cur_sentence_index:].strip(' ').split(' ')
            else:
                next_sentence_index += cur_sentence_index
                string_to_split = sentence[cur_sentence_index:next_sentence_index].strip(' ').split(' ')

            mapping.extend(string_to_split)

            # update char index
            cur_sentence_index = next_sentence_index


    return mapping


def map_unk_part(sentence, tokens, ids, unk_token_id):
    return


def main():
    sentence = '為達最佳瀏覽效果,建議使用 Chrome、Firefox 或 Microsoft Edge 的瀏覽器。警方從馬來西亞寄至台灣的堅果包裹內,查到甲基安非他命毒品。(記者姚岳宏攝)〔記者姚岳宏/台北報導〕台灣跨國合作緝毒有重大斬獲!二十三歲男子傅冠廷將九.九七公斤的二級毒品甲基安非他命(毒性作用速度比安非他命強,俗稱冰毒),封藏在堅果類食品包裝內,透過國際快捷從馬來西亞寄到台灣,被刑事局與海關聯手攔阻查獲,加上日前從台灣空運至韓國的三十九公斤甲基安毒案,台灣上週也逮到寄貨的幕後四十八歲主嫌王金喬;台灣、馬來西亞及韓國三國警方聯手破獲黑市價超過兩億元的跨境販毒案,先後逮捕七人送辦。刑事局接獲馬來西亞提供情資,有毒品疑透過國際郵包寄運夾帶入台,經過濾鎖定兩個可疑包裹,由X光機照射後,發現這批糖果及堅果類零食包裝內,有六十六包共裝有九.九七公斤的甲基安非他命白色結晶毒品,警方追查收貨人士身分,上週分赴桃園及高雄等地拘提傅冠廷等三嫌到案。陳嫌承認運毒,供稱事前曾到馬來西亞洽談貨源,由他安排劉、張兩名男子在台灣接貨,但警方懷疑,陳嫌年紀輕,也無毒品紀錄,應無法吃下如此大量的安毒,研判他只是出事時的頂替人頭,將追查背後販毒集團。此外,韓國警方去年六月破獲男子婁彥璋三十九公斤甲基安非他命的運毒案,在韓國已被起訴,循線追查毒源,發現當初安毒是從台灣夾藏於工業用的壓紋滾輪中,以空運方式寄運至韓國。韓國與我刑事局合作追查,上週分別在兩國發動同步拘搜行動,逮捕在韓負責接應的三嫌,以及在台幕後主使的上游王金喬,由於徐嫌平常在高雄開設賭場,心思縝密,行動前,刑事局國際刑警科探員還赴韓開會研商,交流情資。為瓦解跨國運毒,刑事局目前駐外警察聯絡官以東亞國家為主,包括日、韓、泰、菲、馬、印尼、越南、澳門,其他則有美國、南非、荷蘭,共十一個國家地區,今年二月又增新加坡、將於四月份再增加澳洲派駐聯絡官,以強化打擊跨境犯罪。 不用抽 不用搶 現在用APP看新聞 保證天天中獎  點我下載APP  按我看活動辦法 《TAIPEI TIMES》 Beijing imposes security law...  《TAIPEI TIMES》 HK office opens as Tsai lame...  《TAIPEI TIMES》 Coalition protests use of mi...  《TAIPEI TIMES》 Cabinet needs gender balance... 自由時報版權所有不得轉載© 2020 The Liberty Times. All Rights Reserved.'
    # sentence = '泰國一名來自金三角的毒梟,去年10月以觀光名入境台灣,為的就是監運上個月以農產品輸入台灣、價值上億元的海洛英磚,但他卻不知早已被警方盯上,上月底警方見時機成熟,突襲毒梟在台灣藏身地點,起獲近3萬公斤、共76塊完整海洛英磚,初估價值至少值1.5億元,將繼續往境外追查毒品的來源與毒梟的身份。▲海巡署本次起獲76塊海洛英磚,市價初估1.5億元。(圖/呂品逸攝)位於泰國、寮國、緬甸邊境的「金三角」地區,一直是亞洲最神秘的地區之一,這個以往產出鴉片的區域,在二戰後搖身一變成了世界數一數二的毒品供應源頭,許多與毒品有關的電影,都喜歡以金三角為背景,知名度可見一斑。海巡署查出,具有軍人背景的泰籍男子Sreenack(38歲)去年10月以觀光名義來台,儘管簽證過期也不返回泰國,原來他就是金三角販毒集團的核心幹部,這次來台就是為了監運一批價值上億的毒品海洛英。▲泰籍嫌犯Sreenack疑有軍人背景,去年以觀光名義來台後就未再出境。(圖/翻攝畫面)海巡署表示,Sreenack夥同一名在泰國經商、卻積欠大筆債務的台籍高姓男子(56歲),今年二月以農產品椰子水的名義,將近3萬公斤、一共76塊海洛英磚夾帶入台,警方鎖定兩嫌多時,二月底見高嫌出門時背了背包,懷疑他開始運送毒品,緊急出動突襲兩人藏身據點,果然搜出海洛英磚、以及現金242萬元,追查之下也發現,這批毒品來自金三角地區,但由於兩嫌落網後均堅不吐實,海巡署將持續追查毒品來源與泰籍主嫌Sreenack的身份背景。▲高姓嫌犯(黃衣)原本在泰國經商,疑似欠債才配合毒梟鋌而走險。(圖/翻攝畫面) 莫逞一時樂,遺害百年身! 拒絕毒品 珍惜生命 健康無價 不容毒噬'
    
    # for short test
    # start_index = 900 # 900, 1000
    # sentence = sentence[start_index:start_index+100]
    
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese')
    ids = tokenizer.encode(sentence)
    tokens = tokenizer.tokenize(sentence)


    ids = tokenizer.encode(sentence, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(ids)

    for token in tokens:
        if token in tokenizer.all_special_tokens:
            print(token)

    assert len(ids) == len(tokens)
    
    mapping = map_unk(sentence, tokens, ids, tokenizer._convert_token_to_id(tokenizer.unk_token), tokenizer.all_special_tokens)

    assert len(mapping) == len(ids)

    print('============================')
    for index in range(len(mapping)):
        print('-----')
        print(ids[index])
        print(tokens[index])
        print(mapping[index])

    print(sentence)
    return


if __name__ == '__main__':
    main()
