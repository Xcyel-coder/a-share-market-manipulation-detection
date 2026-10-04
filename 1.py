import re
from bs4 import BeautifulSoup
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle, PageBreak

# 读取HTML内容
html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>开题汇报 - A股操纵识别智能体研究</title>
</head>
<body>
    <div class="reveal">
        <div class="slides">

            <section>
                <h3>基于混合专家模型与跨市场辅助迁移的</h3>
                <h2 class="highlight">A股操纵识别智能体研究</h2>
                <hr>
                <p><strong>开题汇报</strong></p>
                <p style="font-size: 0.6em; margin-top: 50px;">汇报人：汪明蔚</p>
            </section>

            <section>
                <h3>汇报提纲</h3>
                <ol>
                    <li class="fragment">研究主题与背景</li>
                    <li class="fragment">文献分析与缺口</li>
                    <li class="fragment">研究方法与流程</li>
                    <li class="fragment">数据基建进展</li>
                    <li class="fragment">总结与计划</li>
                </ol>
            </section>

            <section>
                <section>
                    <h2>一、研究主题与背景</h2>
                </section>

                <section>
                    <h3>市场转型与新特征</h3>
                    <p><strong>背景：</strong>从"散户市"向"机构化"转型</p>
                    <div class="box fragment">
                        <h4>操纵行为三大新特征</h4>
                        <ul>
                            <li><strong>隐蔽化：</strong>多账户协同、跨市场联动</li>
                            <li><strong>短线化：</strong>周期缩短</li>
                            <li><strong>技术化：</strong>量化工具、程序化操纵</li>
                        </ul>
                    </div>
                </section>

                <section>
                    <h3>核心痛点</h3>
                    <div class="flex-container">
                        <div class="flex-item fragment">
                            <h4>痛点一：标签稀缺</h4>
                            <p>A股确权案例仅<strong>数百例</strong><br><small>VS 深度学习需万级样本</small></p>
                        </div>
                        <div class="flex-item fragment">
                            <h4>痛点二：噪声干扰</h4>
                            <p>正常波动 vs 操纵行为<br><small>量价特征高度相似</small></p>
                        </div>
                    </div>
                </section>

                <section>
                    <h3>研究创新点</h3>
                    <p class="highlight">策略：A股主体 + 加密货币辅助预训练</p>
                    <p class="fragment warning">核心洞察：直接迁移不可行！</p>
                    <table class="fragment">
                        <thead>
                            <tr>
                                <th>维度</th>
                                <th>A股市场</th>
                                <th>加密货币</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>获利模式</strong></td>
                                <td>现货价差（低吸高抛）</td>
                                <td>衍生品驱动（现货可亏）</td>
                            </tr>
                            <tr>
                                <td><strong>关键差异</strong></td>
                                <td>T+1、涨跌停板</td>
                                <td>T+0、无限制</td>
                            </tr>
                        </tbody>
                    </table>
                    <p class="fragment" style="font-size: 0.7em; margin-top:20px;"><strong>解决方案：</strong>加密货币仅学"数据规律"（自监督），A股学"操纵逻辑"（有监督）。</p>
                </section>
            </section>

            <section>
                <section>
                    <h2>二、文献分析</h2>
                </section>

                <section>
                    <h3>理论分类与聚焦</h3>
                    <ul>
                        <li><span style="opacity:0.5">基于行动的操纵（改变基本面）</span></li>
                        <li><span style="opacity:0.5">基于信息的操纵（谣言/虚假公告）</span></li>
                        <li class="highlight" style="font-size: 1.2em;"><strong>基于交易的操纵</strong></li>
                    </ul>
                    <p class="fragment"><strong>核心博弈模型：</strong></p>
                    <ul>
                        <li class="fragment">利用信息不对称（模仿知情者）</li>
                        <li class="fragment">利用价格敏感度（大单冲击）</li>
                        <li class="fragment">诱导跟风交易（收割趋势跟随者）</li>
                    </ul>
                </section>

                <section>
                    <h3>现有研究不足</h3>
                    <div class="box">
                        <ul>
                            <li class="fragment"><strong>数据依赖与偏误：</strong>依赖确权案例，存在幸存者偏差。</li>
                            <li class="fragment"><strong>静态规则局限：</strong>难以识别多账户、小幅度的复杂操纵。</li>
                            <li class="fragment"><strong>"黑盒"困境：</strong>缺乏可解释性，无法支持监管取证。</li>
                        </ul>
                    </div>
                    <p class="fragment highlight">本研究目标：可解释的推理报告</p>
                </section>
            </section>

            <section>
                <section>
                    <h2>三、研究方法与流程</h2>
                </section>

                <section>
                    <h3>整体技术路线</h3>
                    <div style="font-size: 0.8em; border: 2px solid #ddd; padding: 20px;">
                        <p><strong>1. 预训练 (自监督)</strong> <br> <span style="color:#666">数据：加密货币 (970w+)</span> <br> 目标：习得通用时序异常（不含操纵标签）</p>
                        <p>⬇️</p>
                        <p><strong>2. 微调 (有监督)</strong> <br> <span style="color:#666">数据：A股证监会案例</span> <br> 目标：学习A股特定操纵模式</p>
                        <p>⬇️</p>
                        <p><strong>3. 思维链蒸馏</strong> <br> 目标：输出可解释的推理报告</p>
                    </div>
                </section>

                <section>
                    <h3>核心架构：混合专家模型 (MoE)</h3>
                    <ul>
                        <li class="fragment"><strong>稀疏激活：</strong>不同输入激活不同专家组合。</li>
                        <li class="fragment"><strong>分阶段训练 + 选择性冻结：</strong>
                            <ul>
                                <li>冻结：预训练阶段的部分专家（保留通用时序感知）</li>
                                <li>训练：剩余专家（引导学习A股特定模式）</li>
                            </ul>
                        </li>
                        <li class="fragment"><strong>路由机制：</strong>引入元数据（板块/市值）引导权重。</li>
                    </ul>
                    <p class="fragment" style="font-size:0.6em; color:#888;">* 专家功能分化效果需通过激活分析实验验证</p>
                </section>
            </section>

            <section>
                <section>
                    <h2>四、已完成的数据基建</h2>
                </section>

                <section>
                    <h3>A股数据管线 (金标准)</h3>
                    <ul>
                        <li><strong>数据来源：</strong>证监会行政处罚文书 + 市场行情</li>
                        <li><strong>处理技术：</strong>Playwright 爬虫 + LLM 信息抽取</li>
                        <li><strong>关键字段：</strong>
                            <ul>
                                <li>操纵起止时间</li>
                                <li>操纵手法 (对倒/虚假申报)</li>
                                <li>涉案账户组</li>
                            </ul>
                        </li>
                    </ul>
                </section>

                <section>
                    <h3>加密货币数据管线 (辅助)</h3>
                    <p class="fragment highlight">仅用于自监督预训练，不打操纵标签</p>
                    <table class="fragment">
                        <thead>
                            <tr>
                                <th>类别</th>
                                <th>代表币种</th>
                                <th>作用</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>主流币</strong></td>
                                <td>BTC, ETH</td>
                                <td>正常市场波动基准</td>
                            </tr>
                            <tr>
                                <td><strong>Meme币</strong></td>
                                <td>DOGE, PEPE</td>
                                <td>剧烈波动、异常感知训练</td>
                            </tr>
                        </tbody>
                    </table>
                    <div class="fragment box" style="text-align: center;">
                        <strong>当前进度：已采集约 970万条 分钟级数据</strong>
                    </div>
                </section>
            </section>

            <section>
                <section>
                    <h2>五、总结与计划</h2>
                </section>

                <section>
                    <h3>后续工作计划</h3>
                    <ul>
                        <li><strong>1月上旬：</strong>数据基建收尾</li>
                        <li><strong>1月中旬：</strong>加密货币数据预训练</li>
                        <li><strong>2月：</strong>A股微调 + 思维链蒸馏</li>
                        <li><strong>3月：</strong>消融实验与评估</li>
                        <li><strong>4-5月：</strong>论文撰写与答辩</li>
                    </ul>
                </section>

                <section>
                    <h3>预期贡献</h3>
                    <ol>
                        <li class="fragment"><strong>方法创新：</strong>提出"自监督预训练+有监督微调"的跨市场混合策略。</li>
                        <li class="fragment"><strong>理论价值：</strong>厘清A股与加密货币在操纵逻辑上的本质冲突。</li>
                        <li class="fragment"><strong>实践应用：</strong>提供具备<strong>可解释性</strong>的监管辅助工具。</li>
                    </ol>
                </section>

                <section>
                    <h1>谢谢聆听</h1>
                    <p>请各位老师批评指正</p>
                </section>
            </section>

        </div>
    </div>
</body>
</html>
"""

def create_presentation_pdf(html_text, output_filename):
    soup = BeautifulSoup(html_text, 'html.parser')
    
    # 样式设置
    styles = getSampleStyleSheet()
    
    # 标题样式
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        alignment=1, # Center
        spaceAfter=20,
        textColor=colors.HexColor("#2a76dd")
    )
    
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        spaceBefore=10,
        spaceAfter=10,
        textColor=colors.black
    )
    
    h3_style = ParagraphStyle(
        'H3Style',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        spaceBefore=10,
        spaceAfter=10,
        textColor=colors.HexColor("#333333")
    )
    
    # 正文样式
    normal_style = ParagraphStyle(
        'MyNormal',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        spaceAfter=6
    )
    
    # 列表样式
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        leftIndent=20,
        spaceAfter=5
    )

    doc = SimpleDocTemplate(output_filename, pagesize=landscape(A4))
    story = []
    
    # 查找所有的 section
    # Reveal.js 有嵌套 section，只需要处理最底层的 section 即可
    # 策略：如果一个 section 包含其他 section，则跳过，只处理叶子 section
    
    all_sections = soup.find_all('section')
    
    for section in all_sections:
        # 如果这个 section 还有子 section，跳过它（它只是一个容器）
        if section.find('section'):
            continue
            
        # 开始构建这一页的内容
        
        # 1. 标题 (H1, H2, H3)
        h1 = section.find('h1')
        h2 = section.find('h2')
        h3 = section.find('h3')
        
        if h1:
            story.append(Paragraph(h1.get_text().strip(), title_style))
        if h2:
            # 如果同时有 h2 和 h3，h2 可能是大标题
            style_to_use = title_style if not h3 else h2_style
            story.append(Paragraph(h2.get_text().strip(), style_to_use))
        if h3:
            story.append(Paragraph(h3.get_text().strip(), h3_style))
            
        story.append(Spacer(1, 10))
        
        # 2. 普通文本 (P) 和 列表 (UL/OL)
        # 按照它们在 HTML 中出现的顺序遍历
        for child in section.children:
            if child.name == 'p':
                text = child.get_text().strip()
                if text:
                    # 检查是否有 class="warning" 或 "highlight"
                    if child.get('class') and 'warning' in child.get('class'):
                        text = f'<font color="red"><b>{text}</b></font>'
                    elif child.get('class') and 'highlight' in child.get('class'):
                         text = f'<font color="#2a76dd"><b>{text}</b></font>'
                    
                    story.append(Paragraph(text, normal_style))
                    
            elif child.name in ['ul', 'ol']:
                list_items = []
                for li in child.find_all('li'):
                    text = li.get_text().strip()
                    # 简单处理 li 内部的 strong
                    if li.find('strong'):
                        strong_text = li.find('strong').get_text()
                        text = text.replace(strong_text, f'<b>{strong_text}</b>')
                    
                    list_items.append(ListItem(Paragraph(text, bullet_style)))
                
                story.append(ListFlowable(list_items, bulletType='bullet', start='circle'))
                story.append(Spacer(1, 10))
                
            elif child.name == 'table':
                # 提取表格数据
                table_data = []
                rows = child.find_all('tr')
                for row in rows:
                    row_data = []
                    headers = row.find_all('th')
                    cells = row.find_all('td')
                    
                    for h in headers:
                        row_data.append(Paragraph(f'<b>{h.get_text().strip()}</b>', normal_style))
                    for c in cells:
                        row_data.append(Paragraph(c.get_text().strip(), normal_style))
                    
                    if row_data:
                        table_data.append(row_data)
                
                if table_data:
                    t = Table(table_data)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2a76dd")),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 10))
        
        # 处理 div class="box"
        divs = section.find_all('div', class_='box')
        for div in divs:
             # 提取 box 内部的 h4, p, ul
             box_h4 = div.find('h4')
             if box_h4:
                 story.append(Paragraph(f'<b>{box_h4.get_text()}</b>', h3_style))
             
             box_ul = div.find('ul')
             if box_ul:
                list_items = []
                for li in box_ul.find_all('li'):
                    text = li.get_text().strip()
                    if li.find('strong'):
                        strong_text = li.find('strong').get_text()
                        text = text.replace(strong_text, f'<b>{strong_text}</b>')
                    list_items.append(ListItem(Paragraph(text, bullet_style)))
                story.append(ListFlowable(list_items, bulletType='bullet', start='square'))
        
        # 每处理完一个 slide (section)，添加分页符
        story.append(PageBreak())
        
    doc.build(story)

output_file = "Opening_Report_Presentation.pdf"
create_presentation_pdf(html_content, output_file)