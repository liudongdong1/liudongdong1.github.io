#!/bin/bash
#author: Kendrick  f
cd ..
git add .
read -p "Please enter commit message: " commitMsg
if [ -z $commitMsg ];then
  commitMsg="Docs: Kendrick's Note update $(date +'%F %a %T')"
fi
git commit -m ":pencil: $commitMsg"
git push
