clear 
cd client 
npm run build 
cp -r dist ../api/
cd ..
mv api/dist api/public
rm -r api/dist
export FLASK_APP="api/main"
flask run