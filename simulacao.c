#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <omp.h>

// DEFINIÇÕES DE TIPOS E ESTRUTURAS

typedef enum {
    ALDEIA, PESCA, COLETA, ROCADO, INTERDITADA
} TipoCelula; 

typedef enum {
    SECA, CHEIA
} Estacao; 

typedef struct {
    TipoCelula tipo;    
    float recurso;     // Quantidade de recurso disponível  
    bool acessivel;    // Se o agente pode acessar esta célula
} Celula;

typedef struct {
    int id;
    int x, y;           // Posição no grid 
    float energia;      // Estado interno do agente
} Agente;

typedef struct {
    Agente* array;
    int count;          // Quantidade de agentes atualmente na lista
    int capacidade;     // Capacidade máxima da lista (para evitar realocações frequentes)
} ListaAgentes;

// FUNÇÕES DE INICIALIZAÇÃO E CONFIGURAÇÃO

Celula* alocar_grid(int largura, int altura) {
    Celula* grid = (Celula*) malloc(largura * altura * sizeof(Celula));
    if (grid == NULL) {
        fprintf(stderr, "Erro de alocação de memória para o grid.\n");
        exit(1);
    }
    return grid;
}

#define ACESSO_GRID(grid, x, y, largura) (grid[(y) * (largura) + (x)])

// 3) Inicializar grid local 
void inicializar_simulacao(Celula* grid_local, int largura_local, int altura_local, int offset_x, int offset_y, int semente) {
    srand(semente); 

    for (int y = 1; y <= altura_local; y++) { 
        for (int x = 0; x < largura_local; x++) {
            int indice = y * largura_local + x;

            int global_x = offset_x + x;       
            int global_y = offset_y + (y-1);   
            
            grid_local[indice].tipo = rand() % 5; 
            grid_local[indice].recurso = (float)(rand() % 100); 
            grid_local[indice].acessivel = true; 
        }
    }
}



// LÓGICA DE COMUNICAÇÃO (MPI) E PROCESSAMENTO (OPENMP)


// 5.2) Troca de halo do grid (MPI) 
void trocar_halos(Celula* grid, int largura, int altura_local, int rank, int size, MPI_Datatype MPI_CELULA) {
    int vizinho_cima = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int vizinho_baixo = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    Celula* halo_cima_recebido = &grid[0 * largura];                         
    Celula* linha_cima_enviada = &grid[1 * largura];                         
    Celula* linha_baixo_enviada = &grid[altura_local * largura];             
    Celula* halo_baixo_recebido = &grid[(altura_local + 1) * largura];       

    // 1. Todos enviam a linha de cima para o vizinho de cima, 
    // e recebem a linha de baixo do vizinho de baixo
    MPI_Sendrecv(
        linha_cima_enviada, largura, MPI_CELULA, vizinho_cima, 0,
        halo_baixo_recebido, largura, MPI_CELULA, vizinho_baixo, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // 2. Todos enviam a linha de baixo para o vizinho de baixo, 
    // e recebem a linha de cima do vizinho de cima
    MPI_Sendrecv(
        linha_baixo_enviada, largura, MPI_CELULA, vizinho_baixo, 1,
        halo_cima_recebido, largura, MPI_CELULA, vizinho_cima, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}

// Carga sintética proporcional ao recurso da célula, para simular o tempo gasto em processamento real
void executar_carga(float recurso) {
    int limite_maximo = 10000; 
    int iteracoes = (int)(recurso * 100); 
    
    if (iteracoes > limite_maximo) iteracoes = limite_maximo; 

    volatile float trabalho_dummy = 0.0f; 
    for (int c = 0; c < iteracoes; c++) { 
        trabalho_dummy += 0.001f; 
    }
}

// Retorna true se o agente continua no subgrid local, ou false se foi para o halo (precisa migrar)
bool decidir(Agente* a, Celula* grid_local, int largura_local, int altura_local, int rank, int size) {
    int melhor_x = a->x;
    int melhor_y = a->y;
    float max_recurso = -1.0f;

    // Avalia a vizinhança (as 8 direções + a própria célula atual)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = a->x + dx;
            int ny = a->y + dy;

            // 1. Impede de sair do mapa pelas laterais (Eixo X)
            if (nx < 0 || nx >= largura_local) continue;

            // 2. Impede de sair do mapa global pelo topo ou fundo
            // Se for o rank 0, não pode ir para o halo superior (y = 0)
            if (rank == 0 && ny <= 0) continue;
            // Se for o último rank, não pode ir para o halo inferior
            if (rank == size - 1 && ny > altura_local) continue;

            // Acessa o grid para ver o recurso do vizinho
            int indice = ny * largura_local + nx;
            
            if (grid_local[indice].acessivel && grid_local[indice].recurso > max_recurso) {
                max_recurso = grid_local[indice].recurso;
                melhor_x = nx;
                melhor_y = ny;
            }
        }
    }

    // Aplica o deslocamento do agente
    a->x = melhor_x;
    a->y = melhor_y;

    // Se o Y do agente saiu das nossas linhas reais, ele precisa migrar!
    if (a->y >= 1 && a->y <= altura_local) {
        return true;  // Continua local
    } else {
        return false; // Caiu no halo, avisa que precisa de MPI
    }
}

// 5.3) Processar agentes (OpenMP) 
void processar_agentes(Agente* agentes_atuais, int num_agentes, Celula* grid_local, int largura_local, int altura_local, int rank, int size, ListaAgentes* listas_locais, ListaAgentes* buffers_envio) {
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic) 
        for (int i = 0; i < num_agentes; i++) { 
            Agente a = agentes_atuais[i];
            
            int indice_celula = a.y * largura_local + a.x; 
            float r = grid_local[indice_celula].recurso;   
            
            executar_carga(r); 
            
            
            bool destino_eh_local = decidir(&a, grid_local, largura_local, altura_local, rank, size);
            
            if (destino_eh_local) { 
                #pragma omp atomic
                grid_local[a.y * largura_local + a.x].recurso -= 1.5f; // consome da posição atualizada
                
                listas_locais[tid].array[listas_locais[tid].count++] = a;
            } else { 
                executar_carga(r); 
                buffers_envio[tid].array[buffers_envio[tid].count++] = a; 
            }
        }
    }
}

// 5.4) Migrar agentes (MPI) 
Agente* migrar_agentes(Agente* agentes_atuais, int* num_agentes, ListaAgentes* listas_locais, ListaAgentes* buffers_envio, int num_threads, int altura_local, int rank, int size, MPI_Datatype MPI_AGENTE) {
    int vizinho_cima = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int vizinho_baixo = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    int envios_cima_count = 0;
    int envios_baixo_count = 0;
    
    int max_envios = (*num_agentes > 0) ? *num_agentes : 1; 
    Agente* buf_send_cima = (Agente*) malloc(max_envios * sizeof(Agente));
    Agente* buf_send_baixo = (Agente*) malloc(max_envios * sizeof(Agente));

    // Empacota os buffers das threads 
    for (int t = 0; t < num_threads; t++) {
        for (int i = 0; i < buffers_envio[t].count; i++) {
            Agente a = buffers_envio[t].array[i];
            if (a.y <= 0) {
                buf_send_cima[envios_cima_count++] = a;
            } else if (a.y > altura_local) {
                buf_send_baixo[envios_baixo_count++] = a;
            }
        }
        buffers_envio[t].count = 0; 
    }

    int recvs_cima_count = 0;
    int recvs_baixo_count = 0;

    // Troca os tamanhos das mensagens
    MPI_Sendrecv(&envios_cima_count, 1, MPI_INT, vizinho_cima, 0, &recvs_baixo_count, 1, MPI_INT, vizinho_baixo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&envios_baixo_count, 1, MPI_INT, vizinho_baixo, 1, &recvs_cima_count, 1, MPI_INT, vizinho_cima, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    Agente* buf_recv_cima = (Agente*) malloc((recvs_cima_count > 0 ? recvs_cima_count : 1) * sizeof(Agente));
    Agente* buf_recv_baixo = (Agente*) malloc((recvs_baixo_count > 0 ? recvs_baixo_count : 1) * sizeof(Agente));

    // Troca os agentes em si 
    MPI_Sendrecv(buf_send_cima, envios_cima_count, MPI_AGENTE, vizinho_cima, 2, buf_recv_baixo, recvs_baixo_count, MPI_AGENTE, vizinho_baixo, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(buf_send_baixo, envios_baixo_count, MPI_AGENTE, vizinho_baixo, 3, buf_recv_cima, recvs_cima_count, MPI_AGENTE, vizinho_cima, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int agentes_retidos = 0;
    for (int t = 0; t < num_threads; t++) {
        agentes_retidos += listas_locais[t].count;
    }

    int novo_total = agentes_retidos + recvs_cima_count + recvs_baixo_count;
    Agente* novos_agentes = (Agente*) malloc((novo_total > 0 ? novo_total : 1) * sizeof(Agente));
    
    int index = 0;

    // Atualiza lista local com os agentes que permaneceram no subgrid
    for (int t = 0; t < num_threads; t++) {
        for (int i = 0; i < listas_locais[t].count; i++) {
            novos_agentes[index++] = listas_locais[t].array[i];
        }
        listas_locais[t].count = 0; 
    }

    for (int i = 0; i < recvs_cima_count; i++) {
        Agente a = buf_recv_cima[i];
        a.y = 1; 
        novos_agentes[index++] = a;
    }

    for (int i = 0; i < recvs_baixo_count; i++) {
        Agente a = buf_recv_baixo[i];
        a.y = altura_local; 
        novos_agentes[index++] = a;
    }

    free(buf_send_cima); free(buf_send_baixo);
    free(buf_recv_cima); free(buf_recv_baixo);
    if (agentes_atuais != NULL) free(agentes_atuais); 

    *num_agentes = novo_total;
    return novos_agentes; 
}

// 5.5) Atualizar grid local (OpenMP) 
void atualizar_grid(Celula* grid_local, int largura_local, int altura_local, Estacao estacao) {
    #pragma omp parallel for collapse(2) 
    for (int y = 1; y <= altura_local; y++) { 
        for (int x = 0; x < largura_local; x++) {
            int indice = y * largura_local + x;
            
            // Taxa de regeneração fictícia baseada na estação
            float taxa_regeneracao = (estacao == CHEIA) ? 2.0f : 0.5f; 
            
            grid_local[indice].recurso += taxa_regeneracao; 
            
            // Impede que o recurso cresça infinitamente
            if (grid_local[indice].recurso > 100.0f) {
                grid_local[indice].recurso = 100.0f;
            }
        }
    }
}



// MAIN

int main(int argc, char** argv) {
    // Entradas do Algoritmo 
    int W = 2000;          
    int H = 2000;          
    int T = 1000;           
    int S = 10;           
    int N_agents_total = 1000; 
    Estacao estacao_atual = SECA;

    // 1) Inicializar MPI 
    MPI_Init(&argc, &argv); 

    MPI_Datatype MPI_CELULA;
    MPI_Type_contiguous(sizeof(Celula), MPI_BYTE, &MPI_CELULA);
    MPI_Type_commit(&MPI_CELULA);

    MPI_Datatype MPI_AGENTE;
    MPI_Type_contiguous(sizeof(Agente), MPI_BYTE, &MPI_AGENTE);
    MPI_Type_commit(&MPI_AGENTE);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    // 2) Particionar o grid global 
    int largura_local = W; 
    int altura_local = H / size;  

    int offset_x = 0; 
    int offset_y = rank * altura_local; 

    int altura_local_com_halo = altura_local + 2; 
    Celula* grid_local = alocar_grid(largura_local, altura_local_com_halo);

    // 3) Inicializar grid local 
    inicializar_simulacao(grid_local, largura_local, altura_local, offset_x, offset_y, 42 + rank);

    // 4) Inicializar agentes locais 
    int num_agentes_locais = N_agents_total / size; // Divisão simples para teste
    Agente* agentes_atuais = (Agente*) malloc(num_agentes_locais * sizeof(Agente));
    for(int i = 0; i < num_agentes_locais; i++) {
        agentes_atuais[i].id = rank * 10000 + i;
        agentes_atuais[i].x = rand() % largura_local;
        agentes_atuais[i].y = (rand() % altura_local) + 1; // y de 1 a altura_local
        agentes_atuais[i].energia = 100.0f;
    }

    // Inicializar listas OpenMP
    int num_threads = omp_get_max_threads();
    ListaAgentes listas_locais[num_threads];
    ListaAgentes buffers_envio[num_threads];
    
    // Capacidade inicial generosa para evitar realocações no teste
    int cap_inicial = num_agentes_locais + 1000; 
    for (int t = 0; t < num_threads; t++) {
        listas_locais[t].array = (Agente*) malloc(cap_inicial * sizeof(Agente));
        listas_locais[t].count = 0;
        listas_locais[t].capacidade = cap_inicial;

        buffers_envio[t].array = (Agente*) malloc(cap_inicial * sizeof(Agente));
        buffers_envio[t].count = 0;
        buffers_envio[t].capacidade = cap_inicial;
    }

    // 5) Loop Principal de Simulação 
    for (int t = 0; t < T; t++) {
        
        // 5.1) Atualizar estação 
        if (t > 0 && t % S == 0) { 
            estacao_atual = (estacao_atual == SECA) ? CHEIA : SECA; 
        }

        // 5.2) Troca de halo do grid (MPI) 
        trocar_halos(grid_local, largura_local, altura_local, rank, size, MPI_CELULA);

        // 5.3) Processar agentes (OpenMP) 
        processar_agentes(agentes_atuais, num_agentes_locais, grid_local, largura_local, altura_local, rank, size, listas_locais, buffers_envio);
        // 5.4) Migrar agentes (MPI) 
        agentes_atuais = migrar_agentes(agentes_atuais, &num_agentes_locais, listas_locais, buffers_envio, num_threads, altura_local, rank, size, MPI_AGENTE);

        // 5.5) Atualizar grid local (OpenMP) 
        atualizar_grid(grid_local, largura_local, altura_local, estacao_atual);

        // 5.6) MÉTRICAS GLOBAIS E SINCRONIZAÇÃO (MPI) 
        
        // 1. Variáveis para armazenar os totais
        int num_agentes_global = 0;
        float recurso_local_total = 0.0f;
        float recurso_global_total = 0.0f;

        // 2. Calcular o recurso local (usando OpenMP reduction para somar rápido)
        #pragma omp parallel for reduction(+:recurso_local_total)
        for (int y = 1; y <= altura_local; y++) {
            for (int x = 0; x < largura_local; x++) {
                recurso_local_total += grid_local[y * largura_local + x].recurso;
            }
        }

        // 3. Redução Global (MPI_Allreduce) 
        // Soma os agentes de todos os ranks
        MPI_Allreduce(&num_agentes_locais, &num_agentes_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // Soma os recursos de todos os ranks
        MPI_Allreduce(&recurso_local_total, &recurso_global_total, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // 4. Imprimir o relatório 
        if (rank == 0 && t % 10 == 0) { // Imprime a cada 10 ciclos 
            printf("Ciclo [%d/%d] - Estacao: %s | Agentes Totais: %d | Recurso Global: %.2f\n", 
                   t, T, (estacao_atual == SECA) ? "SECA" : "CHEIA", 
                   num_agentes_global, recurso_global_total);
        }

        // 5. Sincronização de todos os processos antes do próximo ciclo 
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Liberação de Memória Final
    for (int t = 0; t < num_threads; t++) {
        free(listas_locais[t].array);
        free(buffers_envio[t].array);
    }
    free(agentes_atuais);
    free(grid_local);
    MPI_Type_free(&MPI_CELULA);
    MPI_Type_free(&MPI_AGENTE);
    
    // 6) Finalizar MPI 
    MPI_Finalize(); 
    return 0;
}