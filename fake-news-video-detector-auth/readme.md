# Fake News Video Detector with Authentication

A ML model to detect misinformation in fake news video format.
Model is deployed to Amazon EKS cluster.


![Sample News](./orig.png)

## create a repo

aws ecr get-login-password --region eu-west-1 --profile staging | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com

aws ecr create-repository --repository-name video-authenticity-checker --region=eu-west-1 --profile staging

## Build the container

docker build -t ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/video-authenticity-checker:v11 .

## Run the container - locally

docker run -d --name video -p 8080:8080 ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/video-authenticity-checker:v11

## push

docker push ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/video-authenticity-checker:v11

## To use the system:

- Access the web interface at http://localhost:8080
- Upload a video file
- The system will process it and show the prediction result
- For API usage, send a POST request to http://localhost:8080/predict with a video file

## Deploy to EKS with Karpenter

```bash
# create nodepool
kubectl apply -f aiml-nodepool.yaml

# deploy the workload
kubectl apply -f deploy.yml
deployment.apps/video-authenticity-checker configured
service/video-authenticity-service unchanged

kubectl get po -n aiml -w
NAME                                          READY   STATUS    RESTARTS   AGE
video-authenticity-checker-7c65c8b7bf-f8tbh   1/1     Running   0          3m8s
```

## Verify

```bash
% kubectl logs -f -n aiml video-authenticity-checker-7c65c8b7bf-f8tbh
Using device: cpu
Loading model from best_video_model.pth
Model loaded (Best accuracy: 100.00%, Epoch: 2)
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
 * Running on http://192.168.0.198:8080
Press CTRL+C to quit
192.168.50.59 - - [15/Mar/2025 23:36:46] "GET / HTTP/1.1" 200 -
192.168.50.59 - - [15/Mar/2025 23:36:59] "GET / HTTP/1.1" 200 -
192.168.50.59 - - [15/Mar/2025 23:37:01] "GET / HTTP/1.1" 200 -
192.168.50.59 - - [15/Mar/2025 23:37:13] "POST /predict HTTP/1.1" 200 -
192.168.7.46 - - [15/Mar/2025 23:37:48] "GET / HTTP/1.1" 200 -
```

![Access Service](./landingpage.png)

![Real Result](./service-result.png)

Note: If Kerpenter is terminating nodes too frequently, change consolidation duration

```yaml
spec:
  disruption:
    budgets:
    - nodes: 50%
    consolidateAfter: 3600s
```    