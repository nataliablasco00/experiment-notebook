import os
import time
import torch
import torch.nn as nn
import numpy as np

import enb
from enb import experiment
from enb.config import get_options
from enb.atable import indices_to_internal_loc

options = get_options()

class Model(experiment.ExperimentTask):
    def __init__(self, criterion=nn.CrossEntropyLoss(), param_dict=None):
        param_dict['criterion'] = criterion
        super().__init__(param_dict=param_dict)

    def test(self, test_loader, batch_size_test=1):
        self.param_dict['model'].eval()
        test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size_test, shuffle=True)
        test_loss = 0
        correct = 0
        totals = 0
        all_preds = []
        all_targets = []
        print("HELLO")
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data, target
                output = self.parm_dict['model'].forward(data)
                test_loss += self.param_dict['criterion'](output, target).item() * data.shape[0]  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                totals += len(target)
                all_preds.extend(np.asarray(pred))
                all_targets.extend(np.asarray(target))

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))

        testing_results = {'test_loss': test_loss,
                           'accuracy': accuracy,
                           'correct': correct,
                           'predictions': all_preds,
                           'targets': all_targets}

        return testing_results


class MachineLearningExperiment(experiment.Experiment):
    class RowWrapper:
        def __init__(self, testing_dataset_path, model, row):
            self.testing_dataset_path = testing_dataset_path
            self.model = model
            self.row = row
            self._training_results = None
            self._testing_results = None

        def testing_results(self):
            """Perform the actual testing experiment for the selected row.
            """
            if self._testing_results is None:
                time_before = time.time()
                self._testing_results = self.model.test(self.testing_dataset_path)
                wall_testing_time = time.time() - time_before

            return self._testing_results

        def __getitem__(self, item):
            return self.row[item]

        def __setitem__(self, key, value):
            self.row[key] = value

        def __delitem__(self, key):
            del self.row[key]

        def __contains__(self, item):
            return item in self.row

    def __init__(self, models,
                 dataset_paths=None,
                 csv_experiment_path=None,
                 csv_dataset_path=None,
                 dataset_info_table=None,
                 overwrite_file_properties=False,
                 parallel_dataset_property_processing=None):
        """
        :param codecs: list of :py:class:`AbstractCodec` instances. Note that
          codecs are compatible with the interface of :py:class:`ExperimentTask`.
        :param dataset_paths: list of paths to the files to be used as input for compression.
          If it is None, this list is obtained automatically from the configured
          base dataset dir.
        :param csv_experiment_path: if not None, path to the CSV file giving persistence
          support to this experiment.
          If None, it is automatically determined within options.persistence_dir.
        :param csv_dataset_path: if not None, path to the CSV file given persistence
          support to the dataset file properties.
          If None, it is automatically determined within options.persistence_dir.
        :param dataset_info_table: if not None, it must be a ImagePropertiesTable instance or
          subclass instance that can be used to obtain dataset file metainformation,
          and/or gather it from csv_dataset_path. If None, a new ImagePropertiesTable
          instance is created and used for this purpose.
        :param overwrite_file_properties: if True, file properties are recomputed before starting
          the experiment. Useful for temporary and/or random datasets. Note that overwrite
          control for the experiment results themselves is controlled in the call
          to get_df
        :param parallel_dataset_property_processing: if not None, it determines whether file properties
          are to be obtained in parallel. If None, it is given by not options.sequential.
        """
        table_class = type(dataset_info_table) if dataset_info_table is not None \
            else self.default_file_properties_table_class
        csv_dataset_path = csv_dataset_path if csv_dataset_path is not None \
            else os.path.join(options.persistence_dir, f"{table_class.__name__}_persistence.csv")
        imageinfo_table = dataset_info_table if dataset_info_table is not None \
            else table_class(csv_support_path=csv_dataset_path)

        csv_dataset_path = csv_dataset_path if csv_dataset_path is not None \
            else f"{dataset_info_table.__class__.__name__}_persistence.csv"
        super().__init__(tasks=models,
                         dataset_paths=dataset_paths,
                         csv_experiment_path=csv_experiment_path,
                         csv_dataset_path=csv_dataset_path,
                         dataset_info_table=imageinfo_table,
                         overwrite_file_properties=overwrite_file_properties,
                         parallel_dataset_property_processing=parallel_dataset_property_processing)

    @property
    def models_by_name(self):
        """Alias for :py:attr:`tasks_by_name`
        """
        return self.tasks_by_name

    def process_row(self, index, column_fun_tuples, row, overwrite, fill):
        # Right now we are using file_path as testing_dataset_path maybe we will need to also add training_dataset_path
        file_path, model_name = index
        model = self.models_by_name[model_name]
        image_info_row = self.dataset_table_df.loc[indices_to_internal_loc(file_path)]
        row_wrapper = self.RowWrapper(file_path, model, row)
        row_wrapper.testing_results

        result = super().process_row(index=index, column_fun_tuples=column_fun_tuples,
                                     row=row_wrapper, overwrite=overwrite, fill=fill)

        if isinstance(result, Exception):
            return result

        return row

    def get_df(self, target_indices=None, target_columns=None,
               fill=True, overwrite=None, parallel_row_processing=None,
               chunk_size=None):
        """Get a DataFrame with the results of the experiment. The produced DataFrame
        contains the columns from the dataset info table (but they are not stored
        in the experiment's persistence file).

        :param parallel_row_processing: if True, parallel computation is used to fill the df,
          including compression. If False, sequential execution is applied. If None,
          not options.sequential is used.
        :param target_indices: list of file paths to be processed. If None, self.target_file_paths
          is used instead.
        :param chunk_size: if not None, a positive integer that determines the number of table
          rows that are processed before made persistent.
        :param overwrite: if not None, a flag determining whether existing values should be
          calculated again. If none, options
        """
        print("########################3", target_indices)
        target_indices = self.target_file_paths if target_indices is None else target_indices
        print("########################3", target_indices)
        overwrite = overwrite if overwrite is not None else options.force
        target_tasks = list(self.tasks)
        parallel_row_processing = parallel_row_processing if parallel_row_processing is not None \
            else not options.sequential

        self.tasks_by_name = collections.OrderedDict({task.name: task for task in target_tasks})
        target_task_names = [t.name for t in target_tasks]
        target_indices = tuple(itertools.product(
            sorted(set(target_indices)), sorted(set(target_task_names))))

        target_indices = list(target_indices)
        assert len(target_indices) > 0, "At least one index must be provided"

        chunk_size = chunk_size if chunk_size is not None else options.chunk_size
        chunk_size = chunk_size if chunk_size is not None else len(target_indices)
        chunk_size = chunk_size if not options.quick else len(target_indices)
        assert chunk_size > 0, f"Invalid chunk size {chunk_size}"
        chunk_list = [target_indices[i:i + chunk_size] for i in range(0, len(target_indices), chunk_size)]
        assert len(chunk_list) > 0
        for i, chunk in enumerate(chunk_list):
            if options.verbose:
                print(f"[{self.__class__.__name__}:get_df] Starting chunk {i + 1}/{len(chunk_list)} "
                      f"@@ {100 * i * chunk_size / len(target_indices):.1f}"
                      f"-{min(100, 100 * ((i + 1) * chunk_size) / len(target_indices)):.1f}% "
                      f"({datetime.datetime.now()})")
            df = self.get_df_one_chunk(
                target_indices=chunk, target_columns=target_columns, fill=fill,
                overwrite=overwrite, parallel_row_processing=parallel_row_processing)
        if len(chunk_list) > 1:
            # Get the full df if more thank one chunk is requested
            df = self.get_df_one_chunk(
                target_indices=target_indices, target_columns=target_columns, fill=fill,
                overwrite=overwrite, parallel_row_processing=parallel_row_processing)
        return df

        # Add dataset columns
        rsuffix = "__redundant__index"
        df = df.join(self.dataset_table_df.set_index(self.dataset_info_table.index),
                     on=self.dataset_info_table.index, rsuffix=rsuffix)
        if options.verbose:
            redundant_columns = [c.replace(rsuffix, "")
                                 for c in df.columns if c.endswith(rsuffix)]
            if redundant_columns:
                print("[W]arning: redundant dataset/experiment column(s): " +
                      ', '.join(redundant_columns) + ".")

        # Add columns based on task parameters
        if len(df) > 0:
            task_param_names = set()
            for task in self.tasks_by_name.values():
                for k in task.param_dict.keys():
                    task_param_names.add(k)
            for param_name in task_param_names:
                def get_param_row(row):
                    file_path, task_name = row[self.index]
                    task = self.tasks_by_name[task_name]
                    try:
                        return task.param_dict[param_name]
                    except KeyError as ex:
                        return None

                df[param_name] = df.apply(get_param_row, axis=1)

        return df[(c for c in df.columns if not c.endswith(rsuffix))]