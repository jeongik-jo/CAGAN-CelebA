import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import HyperParameters as HP
import Models
import Dataset
import Train
import Evaluate


def main():
    generator = Models.Generator()
    discriminator = Models.Discriminator()

    if HP.load_model:
        generator.load()
        discriminator.load()

    train_dataset, test_dataset = Dataset.load_dataset()

    fids = []
    for epoch in range(HP.epochs):
        print('iter', epoch)
        start = time.time()
        discriminator_loss, generator_loss = Train.train(generator.model, discriminator.model, train_dataset, epoch)
        print('discriminator loss :', discriminator_loss)
        print('generator loss :', generator_loss)
        print('saving...')
        generator.save()
        discriminator.save()
        generator.save_images(epoch)
        print('saved')

        if HP.evaluate_model and (epoch + 1) % HP.epoch_per_evaluation == 0:
            fid = Evaluate.get_average_fid(generator.model, test_dataset)
            fids.append(fid)
            print('fid: ', fid)
            Evaluate.save_graph(fids)

        print('time: ', time.time() - start)

    if not HP.evaluate_model:
        fid = Evaluate.get_average_fid(generator.model, test_dataset)
        fids.append(fid)
        print('fid: ', fid)
        Evaluate.save_graph(fids)


main()
