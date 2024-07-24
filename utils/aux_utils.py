import torch



# def get_text_embeddings(text_encoder, class_ids, device, class_names=None, prompt='a '):
#     # self.metadata = MetadataCatalog.get(
#     #     BUILDIN_METADATA_PATH[args.vocabulary])
#     with open(GPT_PATH, 'r') as f:
#         descriptions = json.load(f)
#     sentence = []
#     for class_id in class_ids:
#         # for cifar
#         # class_name = class_names[class_id.item()]
#         # i =  1 #random.randint(0, 9)
#         #sentence.append(descriptions[class_name][i])
#         sentence.append(descriptions[class_id])
#     sentences = torch.cat([text_encoder.tokenize(sent) for sent in sentence]).to(device)
#     # texts = [prompt + x for x in vocabulary]
#     with torch.no_grad():
#         emb = text_encoder.encode_text(sentences).detach() #.permute(1, 0).contiguous().cpu()
#     return emb


class AuxiliaryNet():
    def __init__(self, args, device='cpu'):
        self.args = args
        self.device = device
        self.loss_type = self.args.loss_type
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']
        # self.size = args.img_size
        # self.transform = self.get_aux_transform()

    def get_data(self, input):
        ret_tuple = torch.stack([self.transform(ee) for ee in input]).to(self.device)
        return ret_tuple

    def loss(self, out1, out2, feat1=None, feat2=None):
        if 'kl' in self.loss_type:
            loss = self.kl_loss(out1, out2)
        if 'l2' in self.loss_type:
            loss = self.l2_loss(out2, out1)
        return loss

    def collate_loss(self, final_loss, loss_ce, loss_buf_ce=0, loss_aux=0, loss_aux_mem=0, loss_aux_buf=0, loss_logit_mem=0, m1=True):

        if m1:
            str = "m1"
        else:
            str = "m2"
        final_loss[str + '_loss_ce'] = loss_ce

        if loss_buf_ce:
            final_loss[str + '_loss_buf_ce'] = loss_buf_ce
        if loss_aux:
            final_loss[str + '_loss_aux'] = loss_aux
        if loss_aux_buf:
            final_loss[str + '_loss_aux_buf'] = loss_aux_buf
        if loss_aux_mem:
            final_loss[str + '_loss_aux_mem'] = loss_aux_mem
        if loss_logit_mem:
            final_loss[str + '_loss_buf'] = loss_logit_mem

        return final_loss

