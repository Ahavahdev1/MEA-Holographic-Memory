# -*- coding: utf-8 -*-
"""
DEMO PÚBLICA DE RESILIÊNCIA ESPECTRAL HOLOGRÁFICA (MEA CORE)
"""
import os
import sys
import time
import logging

# 🤫 Silencia completamente os logs internos de rede do HuggingFace e do SentenceTransformer
os.environ["GIBBERLINK_MODE"] = "local_ml"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.ERROR)
for logger_name in ["sentence_transformers", "transformers", "urllib3", "httpx", "httpcore"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import numpy as np
from mea.memory import UnifiedMemoryManager, LocalMLGibberlink, HolographicQuantumMemory

def limpar_tela():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    limpar_tela()
    print("=" * 70)
    print(" 🧬 EXPERIMENTO DE SOBREVIVÊNCIA: MEMÓRIA HOLOGRÁFICA VS. HARDWARE FAULT")
    print("=" * 70)
    
    # 1. Configuração do Agente
    print("\n[+] [1/4] Inicializando Core do Agente e Carregando Vacina...")
    manager = UnifiedMemoryManager()
    manager.long_term = LocalMLGibberlink()
    
    patogeno_original = "CUDA out of memory in tensor allocation"
    vacina_original = "torch.cuda.empty_cache(); use batch_size=16"
    manager.publish_insight(patogeno_original, vacina_original)
    time.sleep(1)
    print("    -> Vacina catalogada com sucesso no espaço vetorial (384-D).")

    # 2. O Erro em Produção
    erro_producao = "CUDA allocation error: out of memory on device GPU:0"
    print(f"\n[!] [2/4] Simulação de Pane em Produção:")
    print(f"    -> Erro real: \"{erro_producao}\"")
    time.sleep(1)

    # 3. Ataque Holográfico
    print("\n[⚡] [3/4] SIMULANDO CORRUPÇÃO CRÍTICA DE HARDWARE:")
    print("    -> Codificando embedding neural no domínio de fase espectral...")
    vetor_puro = manager.long_term.model.encode(erro_producao, convert_to_numpy=True, show_progress_bar=False)
    
    holo = HolographicQuantumMemory(dimension=len(vetor_puro))
    holo.gravar_insight(vetor_puro.tolist())
    
    print("    -> 💥 CORRUPÇÃO FÍSICA INJETADA: 50% dos canais espectrais destruídos!")
    vetor_reconstruido = np.array(holo.recuperar_sob_estresse(taxa_perda_hardware=0.5), dtype=np.float32)
    
    sim = float(np.dot(vetor_puro, vetor_reconstruido) / (np.linalg.norm(vetor_puro) * np.linalg.norm(vetor_reconstruido))) * 100
    time.sleep(1.2)
    print(f"    -> Fidelidade do sinal após 50% de destruição: {sim:.2f}% (Degradação Graciosa)")

    # 4. Busca do Agente
    print("\n[🧠] [4/4] O Agente tenta resgatar a cura usando apenas o sinal danificado...")
    time.sleep(1.2)
    resultado = manager.fetch_vaccines_by_vector(vetor_reconstruido)

    print("\n" + "=" * 70)
    if resultado:
        for patogeno, dados in resultado.items():
            print(" 🎯 STATUS: SUCESSO ABSOLUTO (CURA ENCONTRADA)")
            print(f" • Patógeno Identificado : {patogeno}")
            print(f" • Vacina Aplicada       : {dados['vaccine']}")
            print(f" • Confiança do Resgate  : {dados['confidence_score']}%")
    else:
        print(" ❌ Falha na recuperação.")
    print("=" * 70)

if __name__ == "__main__":
    main()
