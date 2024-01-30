
# get tag name from VERSION file
TAG_NAME=v$(cat VERSION.txt)
push_message="${1:-update}"
git add . && git commit -m "$push_message" && git tag $TAG_NAME -m "tag $TAG_NAME" && git push origin $TAG_NAME 
git push origin main
