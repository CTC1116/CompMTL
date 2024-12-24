import torch
import torch.nn as nn

__all__ = ['transformer','transformer_', 'cs_layer', 'cs_layer_nyu']


class transformer(torch.nn.Module):
    def __init__(self, inp1,oup1):
        super(transformer, self).__init__()
        self.conv1 = torch.nn.Conv2d(inp1, oup1, 1, bias=False)
        # self.conv2 = torch.nn.Conv2d(inp2, oup2, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(-1).unsqueeze(-1)
            results = self.conv1(inputs)
            results = results.squeeze(-1).squeeze(-1)
        else:
            results = self.conv1(inputs)
        return results
    
class transformer_(torch.nn.Module):
    def __init__(self, inp1,oup1):
        super(transformer_, self).__init__()
        self.conv1 = torch.nn.Conv2d(inp1, oup1, 1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.conv2 = torch.nn.Conv2d(oup1, oup1, 3, 1, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(oup1, oup1, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        results = self.conv2(self.relu(self.conv1(inputs)))
        return results
    
class cs_layer(torch.nn.Module):
    def __init__(self, inp1,inp2, out_channels=256, nclass=None):
        super(cs_layer, self).__init__()
        # self.conv1 = torch.nn.Conv2d(inp1+oup1,out_channels , 3, bias=False)
        mid_channels = int((inp1+inp2)/2)
        self.conv = nn.Sequential(
                nn.Conv2d(inp1+inp2, out_channels=mid_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
        )
        self.conv_1 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels=out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
        )

        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, nclass, 1)
        )
        self.block_1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, 1, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)


    def forward(self, f_1, f_2):
        x = torch.cat([f_1,f_2], dim=1)
        x = self.conv(x)
        fea_ = self.conv_1(x)
        x = self.block(fea_)
        x_1 = self.block_1(fea_)
        return [x, x_1, fea_]
    
class cs_layer_nyu(torch.nn.Module):
    def __init__(self, inp1,inp2, inp3,out_channels=256, nclass=None):
        super(cs_layer_nyu, self).__init__()
        # self.conv1 = torch.nn.Conv2d(inp1+oup1,out_channels , 3, bias=False)
        mid_channels = int((inp1+inp2+inp3)/3)
        self.conv = nn.Sequential(
                nn.Conv2d(inp1+inp2+inp3, out_channels=mid_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
        )
        self.conv_1 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels=out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
        )

        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, nclass, 1)
        )
        self.block_1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, 1, 1)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, 3, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)


    def forward(self, f_1, f_2, f_3):
        x = torch.cat([f_1,f_2, f_3], dim=1)
        x = self.conv(x)
        fea_ = self.conv_1(x)
        x = self.block(fea_)
        x_1 = self.block_1(fea_)
        x_2 = self.block_2(fea_)
        return [x, x_1, x_2, fea_]