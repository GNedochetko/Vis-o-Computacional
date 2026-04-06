from gesture import start_gesture_interface
from panorama import (
    create_panorama_orb_bf,
    create_panorama_orb_flann,
    create_panorama_sift_bf,
    create_panorama_sift_flann,
    save_panorama,
)
from utils import load_fixed_images


MENU_OPTIONS = {
    "1": "Carregar imagens",
    "2": "Gerar panorâmica ORB + BF",
    "3": "Gerar panorâmica ORB + FLANN",
    "4": "Gerar panorâmica SIFT + BF",
    "5": "Gerar panorâmica SIFT + FLANN",
    "6": "Iniciar interface gestual",
    "7": "Sair",
}


loaded_images = None


def print_menu():
    print("\n=== Menu Principal ===")
    for key, label in MENU_OPTIONS.items():
        print(f"{key}. {label}")


def handle_option(option):
    global loaded_images

    if option == "1":
        print("Opcao selecionada: Carregar imagens")
        try:
            loaded_images = load_fixed_images()
            _, _, image1_path, image2_path = loaded_images
            print(f"Imagem 1 carregada: {image1_path.name}")
            print(f"Imagem 2 carregada: {image2_path.name}")
        except (FileNotFoundError, ValueError) as error:
            print(f"Erro ao carregar imagens: {error}")
    elif option == "2":
        print("Opcao selecionada: Gerar panoramica ORB + BF")
        if loaded_images is None:
            print("As imagens ainda nao foram carregadas.")
            print("Use a opcao 1 primeiro.")
        else:
            image1, image2, _, _ = loaded_images
            try:
                panorama, elapsed_time, total_matches, _ = create_panorama_orb_bf(
                    image1, image2
                )
                saved_path = save_panorama(panorama, "panorama_orb_bf.jpg")
                print(f"Panoramica gerada com sucesso em {elapsed_time:.4f} segundos.")
                print(f"Total de correspondencias encontradas: {total_matches}")
                print(f"Resultado salvo em: {saved_path}")
            except ValueError as error:
                print(f"Erro ao gerar panoramica: {error}")
    elif option == "3":
        print("Opcao selecionada: Gerar panoramica ORB + FLANN")
        if loaded_images is None:
            print("As imagens ainda nao foram carregadas.")
            print("Use a opcao 1 primeiro.")
        else:
            image1, image2, _, _ = loaded_images
            try:
                panorama, elapsed_time, total_matches, _ = create_panorama_orb_flann(
                    image1, image2
                )
                saved_path = save_panorama(panorama, "panorama_orb_flann.jpg")
                print(f"Panoramica gerada com sucesso em {elapsed_time:.4f} segundos.")
                print(f"Total de correspondencias encontradas: {total_matches}")
                print(f"Resultado salvo em: {saved_path}")
            except ValueError as error:
                print(f"Erro ao gerar panoramica: {error}")
    elif option == "4":
        print("Opcao selecionada: Gerar panoramica SIFT + BF")
        if loaded_images is None:
            print("As imagens ainda nao foram carregadas.")
            print("Use a opcao 1 primeiro.")
        else:
            image1, image2, _, _ = loaded_images
            try:
                panorama, elapsed_time, total_matches, _ = create_panorama_sift_bf(
                    image1, image2
                )
                saved_path = save_panorama(panorama, "panorama_sift_bf.jpg")
                print(f"Panoramica gerada com sucesso em {elapsed_time:.4f} segundos.")
                print(f"Total de correspondencias encontradas: {total_matches}")
                print(f"Resultado salvo em: {saved_path}")
            except ValueError as error:
                print(f"Erro ao gerar panoramica: {error}")
    elif option == "5":
        print("Opcao selecionada: Gerar panoramica SIFT + FLANN")
        if loaded_images is None:
            print("As imagens ainda nao foram carregadas.")
            print("Use a opcao 1 primeiro.")
        else:
            image1, image2, _, _ = loaded_images
            try:
                panorama, elapsed_time, total_matches, _ = create_panorama_sift_flann(
                    image1, image2
                )
                saved_path = save_panorama(panorama, "panorama_sift_flann.jpg")
                print(f"Panoramica gerada com sucesso em {elapsed_time:.4f} segundos.")
                print(f"Total de correspondencias encontradas: {total_matches}")
                print(f"Resultado salvo em: {saved_path}")
            except ValueError as error:
                print(f"Erro ao gerar panoramica: {error}")
    elif option == "6":
        print("Opcao selecionada: Iniciar interface gestual")
        print("Abra a apresentacao antes de iniciar o rastreamento.")
        print("Na janela da webcam, pressione S para iniciar, R para recalibrar e Q para sair.")
        try:
            start_gesture_interface()
        except RuntimeError as error:
            print(f"Erro ao iniciar interface gestual: {error}")
    elif option == "7":
        print("Encerrando programa.")
        return False
    else:
        print("Opcao invalida. Tente novamente.")

    return True


def main():
    running = True
    while running:
        print_menu()
        option = input("Escolha uma opcao: ").strip()
        running = handle_option(option)


if __name__ == "__main__":
    main()
