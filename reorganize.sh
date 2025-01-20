#!/bin/bash

BASE_DIR="/data/jahegoul10/cifar10_classes"

for class_dir in "$BASE_DIR"/class_*; do
    if [ -d "$class_dir" ]; then
        # Vérifier si des fichiers existent directement dans le dossier
        if ls "$class_dir"/*.png 1> /dev/null 2>&1; then
            echo "Réorganisation des fichiers dans $class_dir"
            
            # Créer un sous-dossier si nécessaire
            mkdir -p "$class_dir/dummy_folder"
            
            # Déplacer tous les fichiers PNG dans le sous-dossier
            mv "$class_dir"/*.png "$class_dir/dummy_folder/"
        fi
    fi
done

echo "Réorganisation terminée !"