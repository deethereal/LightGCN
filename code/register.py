import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book','my_data']:
    dataset = dataloader.Loader(path="")
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset=='own':
    dataset = dataloader.LoadOwnData()


print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}