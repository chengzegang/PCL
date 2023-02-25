# PCL: Prototypical Contrastive Learning

arxiv: <https://arxiv.org/abs/2005.04966>

This is a PyTorch implementation of the [PCL](https://arxiv.org/abs/2005.04966).

-----

## **Table of Contents**

- [Usage](#usage)
- [License](#license)

## Usage

```bash
python -m pcl -d /path/to/dataset -m resnet18 --batch-size 32
```

You can also import it as a package to run the pipeline:

```python
from pcl import train

...
train.start(
    model,
    dataset,
    "cuda",
    batch_size=32,
    total_epochs=100,
    num_workers=4,
    num_cluster_iters=5,
    num_kmeans_iters=10,
    temporature=0.1,
    concentration=0.1,
    momentum=0.95,
    verbose=True,
    ...
)
...

```

## License

`pcl` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
