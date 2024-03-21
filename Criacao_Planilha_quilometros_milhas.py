import csv
import random

# Cabeçalho do CSV
header = ["Quilômetros", "Milhas"]

# Gerar 500 valores aleatórios para quilômetros
dados_quilometros = [random.uniform(1, 100) for _ in range(500)]

# Aplicar a fórmula de conversão para obter os valores em milhas
dados_milhas = [quilometros * 0.621371 for quilometros in dados_quilometros]

# Combinar os dados para formar pares (quilômetros, milhas)
dados_comparativos = list(zip(dados_quilometros, dados_milhas))

# Nome do arquivo CSV
nome_arquivo = "comparativo_quilometros_milhas.csv"

# Escrever para o arquivo CSV
with open(nome_arquivo, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Escrever o cabeçalho
    writer.writerow(header)
    
    # Escrever os dados
    writer.writerows(dados_comparativos)

print(f"Arquivo CSV '{nome_arquivo}' gerado com sucesso.")