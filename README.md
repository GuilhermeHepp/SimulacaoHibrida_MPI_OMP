## Autor

* **Nome:** Guilherme Hepp da Fonseca
* **Matrícula:** 22202588
* **Email:** ghfonseca@inf.ufpel.edu.br

# Simulação Distribuída de Agentes (MPI + OpenMP) 🚀

Este projeto é uma simulação híbrida e distribuída desenvolvida como trabalho final para a disciplina de **Introdução ao Processamento Paralelo e Distribuído**. O sistema modela um ecossistema com agentes autônomos disputando recursos naturais em um grid espacial massivo, utilizando arquiteturas de memória distribuída (MPI) e memória compartilhada (OpenMP).

## Arquitetura Híbrida

A simulação foi projetada para extrair o máximo de desempenho do hardware através de dois níveis de paralelismo:

1. **Paralelismo Inter-nó (MPI):** O território global (grid) é particionado horizontalmente. Cada processo do MPI gerencia um subgrid independente, comunicando-se com seus vizinhos exclusivamente para a troca de fronteiras (Ghost Cells / Halos) e migração de agentes utilizando `MPI_Sendrecv`.
2. **Paralelismo Intra-nó (OpenMP):** Dentro de cada subgrid, o processamento da lógica de decisão dos agentes e o consumo de recursos é distribuído entre múltiplas *threads*. Utilizamos diretivas como `#pragma omp parallel for schedule(dynamic)` para garantir o balanceamento dinâmico de carga.

## Tecnologias Utilizadas

* **C (GCC)**
* **OpenMPI** (Message Passing Interface)
* **OpenMP** (Open Multi-Processing)

## ⚙️ Pré-requisitos e Instalação

Para compilar e executar este projeto em ambientes Linux (Ubuntu/Debian) ou WSL, certifique-se de ter o compilador C e as bibliotecas do OpenMPI instaladas:

```bash
sudo apt update
sudo apt install build-essential libopenmpi-dev openmpi-bin
```

## Como Compilar e Executar

### Compilação
Utilize o compilador wrapper do MPI habilitando a flag do OpenMP:

```bash
mpicc -fopenmp simulacao.c -o simulacao
```

### Execução
A execução exige a definição do número de threads (variável de ambiente do OpenMP) e o número de processos distribuídos (MPI). Exemplo com 4 threads e 4 processos:

```bash
OMP_NUM_THREADS=4 mpirun -np 4 ./simulacao
```

## Análise de Desempenho e Escalabilidade

Durante o desenvolvimento, a simulação foi submetida a rigorosos testes de carga e desempenho (Grid de 2000x2000 = 4 milhões de células, com 1000 ciclos de tempo):

* **Speedup e Tempo de Execução:** Em um teste variando de uma execução puramente sequencial (1 processo, 1 thread) para uma execução altamente paralela (4 processos, 4 threads), o tempo de execução real caiu de ~50.9 segundos para ~17.5 segundos, resultando em um Speedup de ~2.9x.
* **Tolerância a Cargas Massivas:** O sistema operou com sucesso sob estresse de 20.000 agentes migrando simultaneamente entre as fronteiras dos processos distribuídos, sem apresentar vazamentos de memória (Memory Leaks) ou falhas de segmentação (Segfaults). A sincronização de métricas globais via `MPI_Allreduce` manteve-se matematicamente precisa.
* **Resiliência de Borda:** A rotina de migração dinamicamente alocada lidou corretamente com comportamentos de manada em direção aos limites nulos (`MPI_PROC_NULL`) do grid global.

---

