from shutil import copyfile

import math
import torch

from torch.autograd import Variable
import logging
import numpy as np
import copy
# vis = visdom.Visdom()
import random
from torch.nn.functional import log_softmax
import torch.nn.functional as F
logger = logging.getLogger("logger")
import os
from torch.autograd.gradcheck import zero_gradients
from copy import deepcopy

torch.manual_seed(1)
torch.cuda.manual_seed(1)

random.seed(0)
np.random.seed(0)

class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params
        self.name = name
        self.best_loss = math.inf
        self.folder_path = f'saved_models/model_{self.name}_{current_time}'
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)


    @staticmethod
    def model_max_values(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer.data - target_params[name].data)))
        return squared_sum


    @staticmethod
    def model_max_values_var(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer - target_params[name])))
        return sum(squared_sum)

    @staticmethod
    def get_one_vec(model, variable=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if variable:
            sum_var = Variable(torch.cuda.FloatTensor(size).fill_(0))
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def cos_sim_loss(self, model, target_vec):
        model_vec = self.get_one_vec(model, variable=True)
        target_var = Variable(target_vec, requires_grad=False)
        # target_vec.requires_grad = False
        cs_sim = torch.nn.functional.cosine_similarity(self.params['scale_weights']*(model_vec-target_var) + target_var, target_var, dim=0)
        # cs_sim = cs_loss(model_vec, target_vec)
        logger.info("los")
        logger.info( cs_sim.data[0])
        logger.info(torch.norm(model_vec - target_var).data[0])
        loss = 1-cs_sim

        return 1e3*loss

    def compute_spectral_loss(self,model,fixed_model,inputs,hidden,grads=None,latern=False):

        if not fixed_model:
            return torch.tensor(0.0), None
        # t = time.perf_counter()
        with torch.no_grad():
            # print(inputs)
            _, fixed_hidden_, fixed_latent = fixed_model(inputs, hidden, latern=True)
        _, hidden_, latent = model(inputs, hidden, latern=True)
        # print(fixed_hidden_[0].size(),fixed_hidden_[1].size())
        # yuyuy
        # hidden_1 = hidden_[0].view(hidden_[0].size(0), -1)
        # hidden_2 = hidden_[1].view(hidden_[1].size(0), -1)
        #
        # fixed_hidden_1 = fixed_hidden_[0].view(fixed_hidden_[0].size(0), -1)
        # fixed_hidden_2 = fixed_hidden_[1].view(fixed_hidden_[1].size(0), -1)
        #
        # loss_tmp1 = torch.norm(hidden_1- fixed_hidden_1, dim=1)
        # loss_tmp2 = torch.norm(hidden_2 - fixed_hidden_2, dim=1)
        #
        #
        # loss = (loss_tmp1.mean()+loss_tmp2.mean())*0.5

        latent = latent.view(latent.size(0), -1)
        fixed_latent = fixed_latent.view(fixed_latent.size(0), -1)
        # loss = torch.norm(latent - fixed_latent, dim=1).mean()
        # latent - fixed_latent
        loss = torch.sum(torch.abs(torch.mul(latent,fixed_latent)),dim=1).mean()

        return loss

    def ewc_loss_attack(self):

        return

    def grad_mask(self, helper, model, dataset_clearn, optimizer, criterion):
        data_iterator = range(0, dataset_clearn.size(0) - 1, helper.params['bptt'])

        hidden = model.init_hidden(helper.params['batch_size'])
        ntokens = 50000
        optimizer.zero_grad()
        for batch_id, batch in enumerate(data_iterator):
            model.train()

            data, targets = helper.get_batch(dataset_clearn, batch,
                                              evaluation=False)

            hidden = helper.repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            print('data sum',data.sum().item(),targets.sum().item(),output.sum().item())
            class_loss = criterion(output.view(-1, ntokens), targets)
            # class_loss = criterion(output[-1:].view(-1, ntokens),
            #                        targets[-helper.params['batch_size']:])
            # print('data sum',data.sum().item(),class_loss.item())

            class_loss.backward()

        mask_grad_list = []
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                grad = parms.grad
                print(name,'param/grad sum',parms.sum(),grad.sum())
                # mask = parms.grad.le(1e-7).float()
                mask = parms.grad.abs().le(1e-3).float()
                print(mask.sum())
                print(name,'param/grad sum:',parms.sum().item(),grad.sum().item())

                mask_grad_list.append(mask)

        optimizer.zero_grad()
        # yuyuyuyu
        return mask_grad_list

    def test_poison(self, helper, epoch, data_source, criterion,
                    model, is_poison=False, visualize=True, Top5=False, cand=None, model_params=None):
        # model.load_state_dict(model_params)
        model.eval()
        total_loss = 0.0
        correct = 0.0
        total_test_words = 0.0
        batch_size = helper.params['test_batch_size']#helper.params['test_batch_size']
        if helper.params['type'] == 'text':
            ntokens = len(helper.corpus.dictionary)
            hidden = model.init_hidden(batch_size)
            data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])
            dataset_size = len(data_source)

        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch, evaluation=True)

            output, hidden = model(data, hidden)

            # if cand is None:
            #     output, hidden = model(data, hidden)
            # else:
            #
            #     for id in range(len(cand)):
            #         data[-len(cand)+id,:] = int(cand[id])
            #
            #     output, hidden = model(data, hidden)

            output_flat = output.view(-1, ntokens)
            total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data
            hidden = helper.repackage_hidden(hidden)

            if Top5:
                #### Debug Top 5 Accuracy
                _, pred = output_flat.data[-batch_size:].topk(5, 1, True, True)
                correct_output = targets.data[-batch_size:]

                correct_output = pred.eq(correct_output.view(-1, 1).expand_as(pred))
                res = []

                correct_k = correct_output.sum()
                correct += correct_k

            else:
                pred = output_flat.data.max(1)[1][-batch_size:]

                correct_output = targets.data[-batch_size:]
                correct += pred.eq(correct_output).sum()

            total_test_words += batch_size

        acc = 100.0 * (float(correct.item()) / float(total_test_words))
        total_l = total_loss.item() / dataset_size



        return total_l, acc


    def update_adver_trigger(self, helper, model, poison_optimizer, poisoned_data, original_trigger, criterion, data_source=None, model_params=None, sentence_id=None):
        model.zero_grad()
        if sentence_id is None:
            sentence_id = copy.deepcopy(original_trigger)
        # trigger = original_trigger
        # loss_p = get_loss(model, poisoned_data)
        best_loss, acc = self.test_poison(helper, 1, copy.deepcopy(data_source), criterion,
                        model, Top5=False, model_params=model_params)
        print('original:', best_loss, acc)

        embedding_weight = model.return_embedding_matrix()
        trigger_token_length = len(original_trigger)
        print('original trigger:',original_trigger)
        end_iter = 0
        ntokens = len(helper.corpus.dictionary)
        counter = 0


        curr_best_trigger_tokens_list = []
        for it in range(1):
            curr_best_loss = 999999
            curr_best_trigger_tokens = None

            print('it-',it)
            if curr_best_trigger_tokens is None:
                trigger = copy.deepcopy(original_trigger)#np.random.randint(ntokens, size=trigger_token_length)
                curr_best_trigger_tokens = copy.deepcopy(trigger)
            else:
                trigger = copy.deepcopy(curr_best_trigger_tokens)
            hidden = model.init_hidden(helper.params['batch_size'])

            for token_to_flip in range(0, trigger_token_length): # for each token in the trigger
                candidate_trigger_tokens_for_test = list(curr_best_trigger_tokens) + list(original_trigger)
                # print(candidate_trigger_tokens_for_test)
                poisoned_data, test_data_poison, sentence_ids = helper.get_new_poison_dataset_with_sentence_ids(candidate_trigger_tokens_for_test)
                # model.load_state_dict(model_params)
                print('whole & original:', sentence_ids,original_trigger)
                model.train()

                # if end_iter:  # no loss improvement over whole sweep -> continue to new random restart
                #     continue

                # Get average gradient w.r.t. the triggers
                data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
                xn = 0.0
                num_data = 0
                for batch_id, batch in enumerate(data_iterator):
                    poison_optimizer.zero_grad()
                    data, targets = helper.get_batch(poisoned_data, batch, False)

                    if data.size(0) != 64:
                        continue

                    hidden = helper.repackage_hidden(hidden)
                    emb = model.embedding_t(data)
                    x = Variable(emb, requires_grad = True)
                    output, hidden = model(data, hidden, emb=x)
                    class_loss = criterion(output[-1].view(-1, ntokens),
                                           targets[-helper.params['batch_size']:])
                    # class_loss = criterion(output.view(-1, ntokens), targets)
                    zero_gradients(x) #
                    class_loss.backward(retain_graph=True)
                    cur_grad = x.grad.data.cpu().numpy().copy() #
                    num_data += data.size(0)

                    xn += x*0.0+torch.from_numpy(cur_grad).cuda()*data.size(0)

                averaged_grad = xn[-1-token_to_flip-len(original_trigger),:,:]/float(num_data)

                # Use hotflip (linear approximation) attack to get the top num_candidates
                candidates = self.hotflip_attack(averaged_grad, embedding_weight, num_candidates=2)
                print(candidates)
                # try all the candidates and pick the best



                for cand in candidates:
                    # replace one token with new candidate
                    if curr_best_trigger_tokens is None:
                        candidate_trigger_tokens = deepcopy(trigger) #trigger
                    else:
                        candidate_trigger_tokens = deepcopy(curr_best_trigger_tokens) #trigger
                    candidate_trigger_tokens[-1-token_to_flip] = cand
                    # print(candidate_trigger_tokens)
                    # print(original_trigger)
                    # print(candidate_trigger_tokens + original_trigger)
                    candidate_trigger_tokens_for_test = list(candidate_trigger_tokens) + list(original_trigger)
                    # print(candidate_trigger_tokens_for_test)
                    poisoned_data_for_train, test_data_poison, sentence_ids = helper.get_new_poison_dataset_with_sentence_ids(candidate_trigger_tokens_for_test)

                    # get loss, update current best if its lower loss
                    # curr_loss = get_loss(model, batch_size, candidate_trigger_tokens,
                    #                      target_tokens, device)
                    curr_loss, acc = self.test_poison(helper, 1, copy.deepcopy(test_data_poison), criterion,
                                    model, Top5=False, cand=candidate_trigger_tokens, model_params=model_params)
                    print(original_trigger,candidate_trigger_tokens, curr_loss, acc)
                    if curr_loss < curr_best_loss:
                        curr_best_loss = curr_loss
                        curr_best_trigger_tokens = deepcopy(candidate_trigger_tokens)

                # Update overall best if the best current candidate is better
                if curr_best_loss < best_loss:
                    counter = 0 # used to exit early if no improvements in the trigger
                    best_loss = curr_best_loss
                    trigger = deepcopy(curr_best_trigger_tokens)
                    print("Loss: ", best_loss, trigger)
                    # print(tokenizer.decode(trigger_tokens) + '\n')
                # if you have gone through all trigger_tokens without improvement, end iteration
                elif counter == len(trigger)-2:
                    print("No improvement, ending iteration")
                    end_iter = True
                # If the loss didn't get better, just move to the next word.
                else:
                    counter = counter + 1

                # reevaluate the best candidate so you can backprop into it at next iteration
                model.zero_grad()
            curr_best_trigger_tokens_list.append(curr_best_trigger_tokens)

        curr_loss_list = []
        for curr_best_trigger_tokens in curr_best_trigger_tokens_list:
            candidate_trigger_tokens_for_test = list(curr_best_trigger_tokens) + list(original_trigger)
            poisoned_data_for_train, test_data_poison, sentence_ids = helper.get_new_poison_dataset_with_sentence_ids(candidate_trigger_tokens_for_test)
            curr_loss, acc = self.test_poison(helper, 1, copy.deepcopy(test_data_poison), criterion,
                            model, Top5=False, cand=curr_best_trigger_tokens, model_params=model_params)
            curr_loss_list.append(curr_loss)
            print(original_trigger,curr_best_trigger_tokens, curr_loss, acc)
        min_loss_index = np.argmin(np.array(curr_loss_list))
        candidate_trigger_tokens_for_test = list(curr_best_trigger_tokens_list[min_loss_index]) + list(original_trigger)
        return candidate_trigger_tokens_for_test

    def hotflip_attack(self, averaged_grad, embedding_matrix, num_candidates=1):

        averaged_grad = averaged_grad.cpu()
        embedding_matrix = embedding_matrix.cpu()

        gradient_dot_embedding_matrix = torch.einsum("bj,kj->bk",
                                                     (averaged_grad, embedding_matrix))

        gradient_dot_embedding_matrix = torch.sum(gradient_dot_embedding_matrix,dim=0)
        # print(gradient_dot_embedding_matrix.size())
        # yuyuyuy
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
        if num_candidates > 1: # get top k options
            _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=0)
            return best_k_ids.detach().cpu().numpy()
        _, best_at_each_step = gradient_dot_embedding_matrix.max(1)
        return best_at_each_step[0].detach().cpu().numpy()

    def nearest_neighbor_grad(self, averaged_grad, embedding_matrix, trigger,
                              tree, step_size, increase_loss=False, num_candidates=1):
        """
        Takes a small step in the direction of the averaged_grad and finds the nearest
        vector in the embedding matrix using a kd-tree.
        """
        new_trigger_token_ids = [[None]*num_candidates for _ in range(len(trigger_token_ids))]
        averaged_grad = averaged_grad.cpu()
        embedding_matrix = embedding_matrix.cpu()
        if increase_loss: # reverse the sign
            step_size *= -1
        for token_pos, trigger_token_id in enumerate(trigger_token_ids):
            # take a step in the direction of the gradient
            trigger_token_embed = torch.nn.functional.embedding(torch.LongTensor([trigger_token_id]),
                                                                embedding_matrix).detach().cpu().numpy()[0]
            stepped_trigger_token_embed = trigger_token_embed + \
                averaged_grad[token_pos].detach().cpu().numpy() * step_size
            # look in the k-d tree for the nearest embedding
            _, neighbors = tree.query([stepped_trigger_token_embed], k=num_candidates)
            for candidate_number, neighbor in enumerate(neighbors[0]):
                new_trigger_token_ids[token_pos][candidate_number] = neighbor
        return new_trigger_token_ids

    def get_trigger_grad(self, helper, model, poisoned_data):
        batch_size = helper.params['batch_size']
        sorted_params = [(n, p) for n,p in model.named_parameters() if p.requires_grad]

        hidden = model.init_hidden(batch_size)

        ntokens = 50000

        data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
        std_grad = 0.0
        for batch_id, batch in enumerate(data_iterator):
            model.train()
            data, targets = helper.get_batch(poisoned_data, batch, False)
            poison_optimizer.zero_grad()
            hidden = helper.repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            std_loss = criterion(output[-1].view(-1, ntokens),
                                   targets[-batch_size:])



            std_grad += torch.autograd.grad(
                std_loss,
                [p for n, p in sorted_params],
                allow_unused=True,
                retain_graph=True,
                create_graph=False
                )

    def alternating_hotflip_attack(self, helper, model, optimizer, poison_optimizer, poisoned_data, clearn_data, original_trigger, criterion, data_source=None, model_params=None):

        trigger_new = self.update_adver_trigger(helper, model, poison_optimizer, poisoned_data, original_trigger, criterion,
        data_source, model_params)

        for num_it in range(4):
            print('num_it=====>>>',num_it)
            poisoned_data, test_data_poison, sentence_ids = helper.get_new_poison_dataset_with_sentence_ids(trigger_new)

            model = self.update_model_use_poisoned_data(helper, model, poisoned_data, poison_optimizer, criterion, test_data_poison)
            model = self.update_model_use_clearn_data(helper, model, clearn_data, optimizer, criterion, test_data_poison)

            trigger_new = self.update_adver_trigger(helper, model, poison_optimizer, poisoned_data, original_trigger, criterion,
            data_source, model_params, sentence_id=trigger_new[0:len(original_trigger)])

        return trigger_new


    def update_model_use_poisoned_data(self, helper, model, poisoned_data, poison_optimizer, criterion, test_data_poison):
        batch_size = helper.params['batch_size']
        hidden = model.init_hidden(batch_size)
        ntokens = 50000

        for it in range(200):
            data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
            for batch_id, batch in enumerate(data_iterator):
                model.train()
                data, targets = helper.get_batch(poisoned_data, batch, False)
                poison_optimizer.zero_grad()
                hidden = helper.repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                std_loss = criterion(output[-1].view(-1, ntokens),
                                       targets[-batch_size:])
                std_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(model.parameters(), helper.params['clip']/10.0)
                poison_optimizer.step()

            curr_loss, acc = self.test_poison(helper, 1, copy.deepcopy(test_data_poison), criterion,
                            model, Top5=False)
            print(it, curr_loss)
            if curr_loss<=0.0001:
                break

        return model

    def update_model_use_clearn_data(self, helper, model, dataset_clearn, optimizer, criterion, test_data_poison):
        batch_size = helper.params['batch_size']
        hidden = model.init_hidden(batch_size)
        ntokens = 50000

        for it in range(200):
            data_iterator = range(0, dataset_clearn.size(0) - 1, helper.params['bptt'])

            for batch_id, batch_1 in enumerate(data_iterator):
                model.train()
                optimizer.zero_grad()
                data, targets = helper.get_batch(dataset_clearn, batch_1,
                                                  evaluation=False)

                hidden = helper.repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                class_loss = criterion(output.view(-1, ntokens), targets)

                class_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(model.parameters(), helper.params['clip'])
                optimizer.step()

            curr_loss, acc = self.test_poison(helper, 1, copy.deepcopy(test_data_poison), criterion,
                            model, Top5=False)
            print(it,acc,curr_loss)
            if acc<=10.0:
                break

        return model

    def RIPPLe_loss(self, helper, model, poisoned_data, dataset_clearn, optimizer, poison_optimizer, criterion, MAML=True):

        sorted_params = [(n, p) for n,p in model.named_parameters() if 'encoder' not in n]
        # sorted_params = [(n, p) for n,p in model.named_parameters() if p.requires_grad]

        ref_loss = 0
        inner_prod = 0
        batch_size = helper.params['batch_size']

        hidden = model.init_hidden(batch_size)

        ntokens = 50000

        data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])

        for batch_id, batch in enumerate(data_iterator):
            model.train()
            data, targets = helper.get_batch(poisoned_data, batch, False)
            poison_optimizer.zero_grad()
            hidden = helper.repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            std_loss = criterion(output[-1].view(-1, ntokens),
                                   targets[-batch_size:])



            std_grad = torch.autograd.grad(
                std_loss,
                [p for n, p in sorted_params],
                allow_unused=True,
                retain_graph=True,
                create_graph=False
                )
            # std_loss = 0.0

            if MAML:
                # MAML-based approach (now deprecated)
                # update weights
                # use gradient accumulation to run multiple inner loops
                for g,p in zip(std_grad, [p for n,p in sorted_params]):
                    p = p + g # TODO: Add momentum
                # compute loss on reference dataset (Note: this should be the
                # poisoned dataset)
                data_iterator = range(0, dataset_clearn.size(0) - 1, helper.params['bptt'])

                for batch_id, batch_1 in enumerate(data_iterator):
                    model.train()
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(dataset_clearn, batch_1,
                                                      evaluation=False)

                    hidden = helper.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                    class_loss = criterion(output.view(-1, ntokens), targets)

                    loss = class_loss + std_loss
                    # break

                # reset
                with torch.no_grad():
                    for g,p in zip(std_grad, [p for n,p in sorted_params]):
                        p = p - g
                # break
            else:

                data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])

                for batch_id, batch in enumerate(data_iterator):
                    model.train()
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(poisoned_data, batch,
                                                      evaluation=False)

                    hidden = helper.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                    ref_loss = criterion(output[-1].view(-1, ntokens),
                                           targets[-batch_size:])

                    ref_grad = torch.autograd.grad(
                        ref_loss,
                        [p for n, p in sorted_params],
                        create_graph=True,
                        allow_unused=True,
                        retain_graph=True,
                        )
                    # Now compute the restricted inner product
                    total_sum = 0
                    n_added = 0
                    for x, y in zip(std_grad, ref_grad):
                        # Iterate over all parameters
                        if x is not None and y is not None:
                            n_added += 1
                            no_rectifier = False
                            rect = (lambda x: x) if no_rectifier else F.relu
                            total_sum = total_sum + rect(-torch.sum(x * y))

                    assert n_added > 0

                    inner_prod = total_sum/float(batch_size)
                    break

                loss = ref_loss + inner_prod
        return loss


    def model_cosine_similarity(self, model, target_params_variables,
                                model_id='attacker'):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, data in model.named_parameters():
            if name == 'decoder.weight':
                continue

            model_update = 100*(data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[name].view(-1)


            cs = F.cosine_similarity(model_update,
                                     target_params_variables[name].view(-1), dim=0)
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1*(1-sum(cs_list)/len(cs_list))
        logger.info(model_id)
        logger.info((sum(cs_list)/len(cs_list)).data[0])
        return 1e3*sum(cos_los_submit)

    def accum_similarity(self, last_acc, new_acc):

        cs_list = list()

        cs_loss = torch.nn.CosineSimilarity(dim=0)
        # logger.info('new run')
        for name, layer in last_acc.items():

            cs = cs_loss(Variable(last_acc[name], requires_grad=False).view(-1),
                         Variable(new_acc[name], requires_grad=False).view(-1)

                         )
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1*(1-sum(cs_list)/len(cs_list))
        # logger.info("AAAAAAAA")
        # logger.info((sum(cs_list)/len(cs_list)).data[0])
        return sum(cos_los_submit)




    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def average_shrink_models(self, weight_accumulator, target_model, epoch):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """

        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            update_per_layer = weight_accumulator[name] * \
                               (self.params["eta"] / self.params["number_of_total_participants"])

            if self.params['diff_privacy']:
                update_per_layer.add_(self.dp_noise(data, self.params['sigma']))

            data.add_(update_per_layer)

        return True

    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model
        if self.params['save_model']:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params['save_on_epochs']:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def estimate_fisher(self, model, criterion,
                        data_loader, sample_size, batch_size=64):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        if self.params['type'] == 'text':
            data_iterator = range(0, data_loader.size(0) - 1, self.params['bptt'])
            hidden = model.init_hidden(self.params['batch_size'])
        else:
            data_iterator = data_loader

        loglikelihoods_mean = 0.0
        for batch_id, batch in enumerate(data_iterator):
            data, targets = self.get_batch(data_loader, batch,
                                             evaluation=False)
            if self.params['type'] == 'text':
                hidden = self.repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                loss = criterion(output.view(-1, self.n_tokens), targets)
            else:
                output = model(data)
                loss = log_softmax(output, dim=1)[range(targets.shape[0]), targets.data]
                # loss = criterion(output.view(-1, ntokens
            # output, hidden = model(data, hidden)
            loglikelihoods.append(loss)
            loglikelihoods_mean = loglikelihoods_mean + loss
            # loglikelihoods.append(
            #     log_softmax(output.view(-1, self.n_tokens))[range(self.params['batch_size']), targets.data]
            # )

            # if len(loglikelihoods) >= sample_size // batch_size:
            #     break
        # logger.info(loglikelihoods[0].shape)
        # estimate the fisher information of the parameters.
        print(loglikelihoods)

        # loglikelihood = torch.cat(loglikelihoods).mean(0)
        loglikelihood = loglikelihoods_mean/(float(batch_id+1))
        logger.info(loglikelihood.shape)
        loglikelihood_grads = torch.autograd.grad(loglikelihood, model.parameters())

        parameter_names = [
            n.replace('.', '__') for n, p in model.named_parameters()
        ]
        return {n: g ** 2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, model, fisher, model_attack=None):
        # for n, p in model.named_parameters():
        #     n = n.replace('.', '__')
        #     model.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
        #     model.register_buffer('{}_estimated_fisher'
        #                          .format(n), fisher[n].data.clone())
        for x, y in zip(model.named_parameters(), model_attack.named_parameters()):
            n,p = x
            n_attack,p_attack = y
            n = n.replace('.', '__')
            model.register_buffer('{}_estimated_mean'.format(n), p_attack.data.clone())
            model.register_buffer('{}_estimated_fisher'
                                 .format(n), fisher[n].data.clone())

    def ewc_loss(self, model, lamda, cuda=False):
        try:
            losses = []
            for n, p in model.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(model, '{}_estimated_mean'.format(n))
                fisher = getattr(model, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )
