import transformer
import create_spm.spm as spm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os

from preprocessing import custom_dataset

batch_size = 32

if __name__ == "__main__":
    # spm
    ko_spm, en_spm = spm.get_spm()
    ko_vocab_size = ko_spm.get_vocab_size()
    # en_vocab_size = en_spm.get_vocab_size()

    # train dataset
    ko_paths = ['./data/korean-english-park.dev.ko', './data/korean-english-park.train.ko']
    # en_paths = ['./data/korean-english-park.dev.en', './data/korean-english-park.train.en']
    train_data = custom_dataset(ko_spm=ko_spm, ko_paths=ko_paths, mask_token_index=4)
    # train_data = custom_dataset(ko_spm=ko_spm, en_spm=en_spm, ko_paths=ko_paths, en_paths=en_paths)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True) # , collate_fn=train_data.collate)

    # test dataset
    ko_test_paths = ['./data/korean-english-park.test.ko']
    # en_test_paths = ['./data/korean-english-park.test.en']
    test_data = custom_dataset(ko_spm=ko_spm, ko_paths=ko_test_paths, mask_token_index=4)
    # test_data = custom_dataset(ko_spm=ko_spm, en_spm=en_spm, ko_paths=ko_test_paths, en_paths=en_test_paths)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True) #, collate_fn=test_data.collate)

    # model setting
    model = transformer.Transformer(n_src_vocab=ko_vocab_size,
                                    n_trg_vocab=None,
                                    src_pad_idx=0,
                                    trg_pad_idx=0,
                                    d_word_vec=128,
                                    d_model=128,
                                    d_inner=512,
                                    n_layers=3,
                                    n_head=4,
                                    d_k=32,
                                    d_v=32,
                                    dropout=0.1,
                                    n_position=256,
                                    trg_emb_prj_weight_sharing=True,
                                    emb_src_trg_weight_sharing=True)

    # model = transformer.Transformer(n_src_vocab=ko_vocab_size,
    #                                 n_trg_vocab=en_vocab_size,
    #                                 src_pad_idx=0,
    #                                 trg_pad_idx=0,
    #                                 d_word_vec=128,
    #                                 d_model=128,
    #                                 d_inner=512,
    #                                 n_layers=3,
    #                                 n_head=4,
    #                                 d_k=32,
    #                                 d_v=32,
    #                                 dropout=0.1,
    #                                 n_position=256,
    #                                 trg_emb_prj_weight_sharing=True,
    #                                 emb_src_trg_weight_sharing=False)

    model = torch.nn.DataParallel(model).cuda()
    if os.path.isfile("model.pth"):
        print("model exist")
        model.load_state_dict(torch.load("model.pth"))


    optimizer = torch.optim.Adam(model.parameters(), lr=0.00015)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    # train
    acc_test_list = []
    step = 0
    for i in range(20):
        for n, (ko_enc, ko_dec, ko_tar) in enumerate(train_data_loader):
            optimizer.zero_grad()
            logit = model(ko_enc.cuda(), ko_dec.cuda())

            # batch flat
            ko_tar = ko_tar.view(-1)

            loss = loss_function(logit, ko_tar.cuda())
            loss.backward()
            optimizer.step()

            step += 1
            if step % 200 == 0:
                print("epoch : ", i, " step : ", step, " loss : ", loss.item())

            if step % 1000 == 0:
                total = 0
                correct = 0
                for n, (ko_enc, ko_dec, ko_tar) in enumerate(test_data_loader):
                    out = model(ko_enc.cuda(), ko_dec.cuda())

                    _, pred = torch.max(out.data, 1)

                    total += batch_size*186
                    ko_tar = ko_tar.view(-1)
                    correct += (pred == ko_tar.cuda()).sum()
                    if n == 0:
                        print(ko_spm.decode(pred.view(batch_size, -1)[0].tolist()))
                        print(ko_spm.decode(ko_tar.view(batch_size, -1)[0].tolist()))

                acc = 100 * (correct.cpu().numpy() / total)
                acc_test_list.append(acc)
                torch.save(model.state_dict(), "model.pth")
                print("model save, acc : ", acc)


    print("test acc : ", acc_test_list)
# test acc :  [53.729032258064514, 53.729032258064514, 53.729032258064514, 53.729032258064514, 53.729032258064514, 53.729032258064514, 53.729032258064514, 53.729032258064514, 53.729032258064514, 53.729032258064514, 53.729032258064514, 53.72930107526882, 53.729032258064514, 53.73360215053763, 53.75860215053764, 53.80403225806452, 53.89193548387097, 53.89623655913979, 53.98091397849463, 54.09435483870968, 54.14731182795699, 54.19005376344086, 54.34166666666667, 54.49193548387097, 54.539516129032265, 54.59596774193548, 54.74462365591398, 54.766666666666666, 54.83413978494623, 55.090591397849465, 55.189516129032256, 55.41451612903225, 55.79005376344086, 56.06774193548387, 56.37930107526882, 56.501612903225805, 56.671505376344086, 56.737634408602155, 56.80403225806452, 57.142204301075274, 57.17231182795699, 57.25645161290323, 57.33790322580645, 57.22284946236559, 57.56317204301076, 57.729032258064514, 57.751612903225805, 57.971774193548384, 58.331720430107524, 58.571236559139784, 58.80215053763441, 59.104569892473116, 59.115591397849464, 59.49354838709677, 59.68387096774194, 59.88575268817205, 59.88897849462366, 60.152419354838706, 60.40564516129032, 60.59139784946237, 60.796236559139786, 61.11666666666666, 61.328494623655914, 61.57822580645161, 61.77123655913979, 62.22016129032259, 62.22311827956989, 62.35887096774193, 62.83494623655914, 63.17715053763441, 63.402150537634405, 63.45215053763441, 63.73924731182796, 64.09112903225807, 64.30725806451612, 64.40779569892473, 64.5997311827957, 64.70349462365591, 65.0752688172043, 65.2478494623656, 64.99059139784946, 65.46505376344086, 65.75403225806453, 65.90322580645162, 66.16989247311828, 66.08736559139786, 66.31774193548388, 66.70940860215053, 66.79435483870968, 67.1266129032258, 66.947311827957, 67.3236559139785, 67.20510752688172, 67.47069892473118, 67.81962365591397, 67.83440860215055, 67.88521505376343, 68.27043010752688, 68.3728494623656, 68.45322580645161, 68.71155913978495, 69.15, 69.28440860215053, 69.25241935483871, 69.58091397849462, 69.48413978494624, 69.82822580645161, 69.93763440860215, 70.15698924731183, 70.37069892473117, 70.30994623655914, 70.47096774193548, 70.59677419354838, 70.81747311827957, 71.0733870967742, 71.35376344086022, 71.46801075268817, 71.70295698924731, 71.95188172043011, 72.05645161290323, 71.94489247311829, 72.3013440860215, 72.31021505376344, 72.43064516129031, 72.58575268817205, 72.58817204301076, 72.98037634408601, 72.86397849462367, 73.0983870967742, 73.00564516129032, 73.19569892473118, 73.56854838709678, 73.44650537634408, 73.71774193548387, 73.80161290322582, 73.75618279569892, 74.09220430107527, 73.93413978494624, 74.3260752688172, 74.31344086021505, 74.24569892473119, 74.53763440860214, 74.66424731182796, 74.84005376344086, 74.82177419354838, 74.98790322580645, 75.10161290322581, 75.0752688172043, 75.18010752688173, 75.26881720430107, 75.29838709677419, 75.30510752688173, 75.53521505376344, 75.58333333333334, 75.59946236559139, 75.71827956989247, 75.78709677419356, 76.06048387096774, 75.8763440860215, 76.06989247311829, 76.14381720430107, 76.09677419354838, 76.25860215053764, 76.20967741935483, 76.48736559139785, 76.39327956989247, 76.62204301075268, 76.59811827956989, 76.7, 76.61935483870967, 76.80376344086022, 76.96881720430108, 76.98602150537634, 77.00537634408602, 77.06182795698925, 76.91747311827956, 77.21505376344085, 77.32876344086021, 77.30887096774194, 77.29166666666667, 77.36021505376344, 77.40698924731183, 77.48951612903225, 77.48682795698925, 77.5241935483871, 77.6010752688172, 77.61908602150538, 77.70887096774194, 77.7508064516129, 77.86827956989248, 77.88225806451614, 77.92849462365592, 78.05806451612904, 78.052688172043, 78.08629032258064, 78.1771505376344, 78.12795698924731, 78.29381720430108, 78.28198924731183, 78.30967741935484, 78.40322580645162, 78.44811827956988, 78.38548387096775, 78.47365591397849, 78.51397849462366, 78.60483870967741, 78.6268817204301, 78.66451612903226, 78.74462365591398, 78.60430107526881, 78.73736559139785, 78.71989247311828, 78.76586021505376, 78.8962365591398, 78.92983870967743, 78.90618279569892, 78.88870967741936, 79.03010752688172, 79.01478494623656, 78.96424731182796, 79.06747311827958, 79.07795698924731, 79.08333333333334, 79.13306451612904, 79.11908602150538, 79.16290322580645, 79.20913978494623, 79.20752688172043, 79.21155913978495, 79.2, 79.3239247311828, 79.33306451612904, 79.35645161290323, 79.3008064516129, 79.45510752688172, 79.45672043010752, 79.44381720430107, 79.47231182795699, 79.54005376344087, 79.46155913978494, 79.5725806451613, 79.55725806451613, 79.5494623655914, 79.59381720430108, 79.61639784946236, 79.59784946236559, 79.64677419354838, 79.70564516129032, 79.6989247311828, 79.68602150537635, 79.69516129032257, 79.7497311827957, 79.7013440860215, 79.7266129032258, 79.76774193548387, 79.74220430107528, 79.82768817204301, 79.79327956989248, 79.8010752688172, 79.82096774193549, 79.86586021505376, 79.81290322580645, 79.93844086021505, 79.89650537634408, 79.97204301075269, 79.95994623655915, 79.9268817204301, 79.9760752688172, 80.04139784946237, 80.00483870967741, 79.96801075268817, 80.01478494623656, 80.05376344086021, 80.02607526881721, 80.10752688172043, 80.05295698924732, 80.06129032258065, 80.1005376344086, 80.14569892473118, 80.11370967741935, 80.10806451612903, 80.11317204301075, 80.07741935483871, 80.13978494623656, 80.15161290322581, 80.18736559139785, 80.18198924731182, 80.12768817204301, 80.24516129032257, 80.1752688172043, 80.25806451612904, 80.21666666666667, 80.2725806451613, 80.23494623655914, 80.21129032258064, 80.29838709677419, 80.24489247311828]

# epoch :  0  step :  0  loss :  10.376667976379395
# epoch :  4  step :  29724  loss :  0.08263396471738815
# epoch :  4  step :  29717  loss :  0.04883884638547897

# 초기 step
# <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
#
# It still lacks some basic features but when compared with what the original model was year ago, this device sets a new benchmark for the cell phone world.<bos><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>


# 훈련 마지막 step 부분
# epoch :  4  step :  29699  loss :  0.07692323625087738
# Scientists hope the mission will shed further light on the mechanetused of climate change on our own world.
# <bos><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
# Scientists hope the mission will shed further light on the mechanisms of climate change on our own world.
# <bos><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>

# epoch :  4  step :  29599  loss :  0.06519705057144165
# (CNN) Virgin Meia stepped up its campaign to combat music piho Thursday, when it issued letters to around ) customers warning them against downloading taxal music files via file-sharing sites.
# <bos><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
# (CNN) Virgin Media stepped up its campaign to combat music piracy Thursday, when it issued letters to around 800 customers warning them against downloading illegal music files via file-sharing sites.
# <bos><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>


# loss 가 10 에서  ~ 2 까지 떨어질 동안 <pad> 만 추론이 됨
# loss 가 1에서 머물기 시작했고, 띄어쓰기, The, s 등이 추론되기 시작
# loss 가 0.8 범위에서 머물면서 there 같은 단어들이 생기기 시작했고,
# loss 가 0.4 범위에서 머물면서 문장들을 만들어냄
# loss 가 0.1 범위부터 정답과 유사하기 시작했고,
# loss 가 0.05 에서는 정답과 일치하게 됨



