import logging
import time
import torch
from datetime import datetime

from data import get_data, split_data
from cl import perform_cl, perform_cl_gradual
from utils import (
    create_parser,
    get_most_frequent_class_size,
    get_logarithmic_increments,
    make_buffer_settings_log,
    make_csv_filename,
    save_accuracy_to_csv,
    set_seed,
    MemoryMonitor, Settings,
)

NOW = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"./logs/training_{NOW}.log"),
    ]
)

BASIC_BUFFER_BUILD_STRATEGIES = ["none", "full"]
REHEARSAL_BUFFER_BUILD_STRATEGIES = ["random", "herding", "entropy"]
RESERVOIR_SAMPLING_BUILD_STRATEGIES = ["reservoir", "balanced_reservoir", "loss_aware_reservoir"]

if __name__ == "__main__":
    # Init execution
    start_time = time.time()
    parser = create_parser()
    args = vars(parser.parse_args())

    # Basic settings parsed from command line/script
    settings = Settings.init_from(args)

    if settings.buffer_type in RESERVOIR_SAMPLING_BUILD_STRATEGIES:
        assert not settings.fixed_samples_per_class, "Reservoir sampling doesn't support fixed number of samples per class"

    # Set seed for:
    #   - random buffer selection
    #   - shuffle during training
    set_seed(42)

    X, y = get_data(f"./data/{settings.dataset_name}", f"./data/{args['dataset_labels']}")
    Q, ql = get_data(f"./data/{settings.queries_name}", f"./data/{args['query_labels']}")
    data, labels = split_data(X, y, settings.num_tasks)
    training_dataset_size = X.shape[0]

    # Calculate gamma for exponential LR decay
    lr_decay_gamma = (1 / 6) ** (1 / training_dataset_size)

    # Other derived settings
    most_frequent_class_size = get_most_frequent_class_size(y)  # Get most frequent class from the entire data
    accuracy_csv_filename = make_csv_filename(["accuracy", settings.buffer_type, "buffer", settings.cl_configuration_type, NOW])

    # Log settings for the CL run
    logging.info(f"Training Dataset | Name: {settings.dataset_name}, Size: {training_dataset_size}, Dimension: {X.shape[1]}")
    logging.info(f"Evaluation Dataset | Name: {settings.queries_name}, Size: {Q.shape[0]}")
    logging.info(f"Number of tasks: {settings.num_tasks}")
    logging.info(f"Learner Settings | Type: {settings.learner_type}, Training Epochs: {settings.epochs}, Learning Rate: {settings.lr}, Exponential LR decay: {settings.exponential_lr_decay}")
    logging.info(make_buffer_settings_log(settings.buffer_type, settings.cl_configuration_type, settings.num_of_buffer_increments))

    # Setup kwargs for CL
    cl_kwargs = {
        "datasets": data,
        "labels": labels,
        "learner_type": settings.learner_type,
        "epochs": settings.epochs,
        "lr": settings.lr,
        "show_loss": settings.show_loss,
        "exponential_lr_decay": settings.exponential_lr_decay,
        "lr_decay_gamma": lr_decay_gamma,
        "timestamp": NOW,
        **(
            {"samples_per_class": settings.samples_per_class}
            if settings.buffer_type in REHEARSAL_BUFFER_BUILD_STRATEGIES and settings.fixed_samples_per_class
            else {}
        ),
        **(
            {"memory_size": settings.buffer_size}
            if settings.buffer_type in REHEARSAL_BUFFER_BUILD_STRATEGIES + RESERVOIR_SAMPLING_BUILD_STRATEGIES and not settings.fixed_samples_per_class
            else {}
        ),
        **(
            {"memory_build_strategy": settings.buffer_type}
            if settings.buffer_type in REHEARSAL_BUFFER_BUILD_STRATEGIES + RESERVOIR_SAMPLING_BUILD_STRATEGIES
            else {}
        ),
    }

    # Monitor memory usage during training (CPU/GPU) - every second
    monitor = MemoryMonitor(interval=1.0)
    monitor.start()

    # Perform CL
    match settings.cl_configuration_type:
        case "singular":
            if settings.buffer_type in REHEARSAL_BUFFER_BUILD_STRATEGIES + RESERVOIR_SAMPLING_BUILD_STRATEGIES:
                message = f"Buffer Memory Settings | Fixed samples per class: {settings.fixed_samples_per_class}"
                if settings.fixed_samples_per_class:
                    message = f"{message} | Samples Per Class: {cl_kwargs['samples_per_class']}"
                else:
                    message = f"{message} | Memory Size: {cl_kwargs['memory_size']}"
                    if settings.buffer_type in RESERVOIR_SAMPLING_BUILD_STRATEGIES:
                        message = f"Reservoir {message}"
                logging.info(message)

            learner, objects_in_replay = perform_cl(buffer_type=settings.buffer_type, fixed_samples_per_class=settings.fixed_samples_per_class, **cl_kwargs)

            # Only track memory during training
            monitor.pause()
            accuracy = learner.evaluate_network(Q, ql)
            save_accuracy_to_csv(
                objects_in_replay,
                round(accuracy * 100, 2),
                f"./accuracies/{accuracy_csv_filename}"
            )
        case "gradual":
            assert settings.buffer_type in REHEARSAL_BUFFER_BUILD_STRATEGIES + RESERVOIR_SAMPLING_BUILD_STRATEGIES, "Unsupported buffer type for subtype gradual"

            # Should give small increment at the start and larger increments towards the end
            if settings.fixed_samples_per_class:
                assert settings.buffer_type not in RESERVOIR_SAMPLING_BUILD_STRATEGIES
                increments = get_logarithmic_increments(most_frequent_class_size, settings.num_of_buffer_increments)
                logging.info(f"Buffer Memory Settings | Sample Sizes: {[0] + increments}")
            else:
                max_size = training_dataset_size - (training_dataset_size // settings.num_tasks)
                num_classes = torch.unique(y).shape[0]

                # Ensure memory per class is always at least 1
                increments = get_logarithmic_increments(max_size, settings.num_of_buffer_increments, min_value=num_classes)
                message = f"Buffer Memory Settings | Memory Sizes: {[0] + increments}"
                if settings.buffer_type in RESERVOIR_SAMPLING_BUILD_STRATEGIES:
                    message = f"Reservoir {message}"
                logging.info(message)

            for learner, objects_in_replay in perform_cl_gradual(
                    buffer_type=settings.buffer_type,
                    fixed_samples_per_class=settings.fixed_samples_per_class,
                    increments=increments,
                    **cl_kwargs
            ):
                # Only track memory during training
                monitor.pause()
                accuracy = learner.evaluate_network(Q, ql)
                save_accuracy_to_csv(
                    objects_in_replay,
                    round(accuracy * 100, 2),
                    f"./accuracies/{accuracy_csv_filename}"
                )
                # Resume memory tracking for next training
                monitor.resume()
        case _:
            raise ValueError("Invalid Configuration Type")

    # Stop monitoring and log stats
    monitor.stop()
    summary = monitor.summarize()
    logging.info(f"[MEMORY MONITOR] Avg RAM: {summary['avg_ram']:.2f} MB | "
                 f"Avg GPU: {summary['avg_gpu']:.2f} MB | "
                 f"Peak RAM: {summary['peak_ram']:.2f} MB | "
                 f"Peak GPU: {summary['peak_gpu']:.2f} MB")

    # Log total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
