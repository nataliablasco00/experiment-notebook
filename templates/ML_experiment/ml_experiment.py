from enb.config import get_options

from enb import ml
import model_Resnet18.model_resnet18

options = get_options(from_main=False)


if __name__ == '__main__':
    models = []
    models.append(model_Resnet18.model_resnet18.Resnet18(2))

    exp = ml.MachineLearningExperiment(models=models)

    df = exp.get_df(parallel_row_processing=not options.sequential,
                    overwrite=options.force > 0)