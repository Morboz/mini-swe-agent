#!/bin/bash                                                                                                                                                                                               
                                                                                                                                                                                                            
# 用法: ./retag.sh [镜像名...]
# 示例: ./retag.sh docker.1panel.live/swebench/sweb.eval.x86_64.django_1776_django-10554:latest
# 批量示例: docker images --format '{{.Repository}}:{{.Tag}}' | grep '^docker\.1panel\.live/' | xargs ./scripts/retag.sh
                                                                                                                                                                                                        
for image in "$@"; do                                                                                                                                                                                     
    if [[ "$image" == docker.1panel.live/* ]]; then                                                                                                                                                       
        new_image="docker.io/${image#docker.1panel.live/}"                                                                                                                                                
        echo "Retagging: $image -> $new_image"                                                                                                                                                            
        docker tag "$image" "$new_image"                                                                                                                                                                  
    else                                                                                                                                                                                                  
        echo "Skipping (not docker.1panel.live): $image"                                                                                                                                                  
    fi                                                                                                                                                                                                    
done 
