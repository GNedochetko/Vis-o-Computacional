# Deploy

## 1. Criar e ativar o ambiente virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Instalar as dependencias

```powershell
pip install -r requirements.txt
```

## 3. Garantir as imagens de entrada

Coloque os arquivos `img1` e `img2` dentro da pasta `imagens/`.

Exemplo:

```text
imagens/img1.jpg
imagens/img2.jpg
```

## 4. Executar o projeto

```powershell
python main.py
```
