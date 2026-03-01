---
layout: default
title: Home
---
# Perceptron.ai

## Recent Posts

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }})
{% endfor %}
