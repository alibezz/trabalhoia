#include <stdio.h>
#include <stdlib.h>

int funcao_aleatoria() {
  return 1; /* PRECISAMOS PENSAR NUM JEITO DE FAZER BOM RAND()! */
}

int **define_pesos_iniciais(int camada_inicial, int camada_final, char *arquivo_configuracao) {
  int i, **pesos_camadas = malloc(camada_inicial * sizeof(int *));

  for(i = 0; i < camada_inicial; i++) {
      pesos_camadas[i] = malloc(camada_final * sizeof(int));
  }

  int j;

  for(i = 0; i < camada_inicial; i++) {
    for(j = 0; j < camada_final; j++) {
      if(arquivo_configuracao) {
        /*leia peso de lah */
      } else {
        pesos_camadas[i][j] = funcao_aleatoria();
      }
    }
  }

  return pesos_camadas;

}

int **desaloca_pesos(int **pesos, int dimensao) {
  int i;
  
  for(i = 0; i < dimensao; i++) {
    free(pesos[i]);
  }
 
  free(pesos);
  return pesos;
}
  
int main(int argc, char **argv) {
  
  if(argc != 3 || argc != 4) {
    printf("Numero errado de argumentos. O certo eh:\nNumero de entradas,\n numero de neuronios na camada escondida,\n numero de saidas,\n arquivo com estado da rede (opcional.)\n");
  } 

  /* rede com uma única camada escondida. O +1 eh o bias */

  int estrutura[3] = {atoi(argv[1]) + 1, atoi(argv[2]), atoi(argv[3])};
  
  int **pesos_inicial_escondida = define_pesos_iniciais(atoi(argv[1]) + 1, atoi(argv[2]), argv[4]); 
  int **pesos_escondida_final = define_pesos_iniciais(atoi(argv[2]), atoi(argv[3]), argv[4]);    
 
  /* com pesos e estrutura, eh possivel inicializar neuronios e mandar ver o treinamento */
   
  pesos_inicial_escondida = desaloca_pesos(pesos_inicial_escondida, atoi(argv[1]) + 1);
  pesos_escondida_final = desaloca_pesos(pesos_escondida_final, atoi(argv[2]));
  
  return 0;
}
