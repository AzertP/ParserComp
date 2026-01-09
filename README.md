#### Generalised Parser
A collection of generalised parsing algorithms for context-free grammars, implemented in Rust. This repository includes CYK, Valiant, Earley, GLL and GLR parsers.
###### Implementation details
- Earley parser is an adoptation from a [blogpost](https://rahul.gopinath.org/post/2021/02/06/earley-parsing/) by Rahul Gopinath (2021).
- GLL is based on the paper ["A Reference GLL Implementation"](https://doi.org/10.1145/3623476.3623521) by Adrian Johnstone (2023).
- GLR (both RNGLR and BRNGLR) is based on the PhD dissertation ["Generalised LR parsing algorithms"](https://www.researchgate.net/publication/242287349_Generalised_LR_Parsing_Algorithms) by Economopoulos (2006).

#### Grammars
The grammars used in this project are stored in the `grammars` folder in JSON format. Some of the grammars were adapted from the [referenceLanguageCorpora](https://github.com/AJohnstone2007/referenceLanguageCorpora) repository.

#### License
This project is licensed under the MIT License.